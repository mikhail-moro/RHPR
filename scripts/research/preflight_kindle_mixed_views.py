"""CPU-only preflight on pinned real inputs and the native GRID generator."""
import argparse
import copy
import json
import tempfile
from pathlib import Path

import numpy as np
import torch

from rhpr.gr.data import file_sha256
from rhpr.gr.experiment import _loss
from rhpr.gr.grid import build_grid_generator
from rhpr.gr.kindle_mixed_views import MixedViews, validate
from rhpr.gr.kindle_pairs import encode
from rhpr.gr.kindle_windows import encode_window
from rhpr.gr.ml100k_pair_raw import cases
from rhpr.gr.pair_unique4 import InputIndex4, attach4
from rhpr.gr.sid import SidIndex
from rhpr.gr.sid_study import state_digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--parent',type=Path,required=True)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--config',type=Path,default=Path('experiments/kindle_mixed_views.json'))
    parser.add_argument('--resume',type=Path)
    args = parser.parse_args()
    c = json.loads(args.config.read_text()); validate(c)
    torch.set_num_threads(1)
    for name,expected in c['parent']['files'].items():
        path = args.parent/name
        assert path.stat().st_size == expected['size_bytes'] and file_sha256(path) == expected['sha256'],name
    for name,sha in c['movie_input']['sha256'].items():
        assert file_sha256(args.data/name) == sha,name
    old = json.loads((args.parent/'run-manifest.json').read_text())['config']
    for key in ('training','generator','data','movie_input','quantizer','evaluation'):
        assert c[key] == old[key],key
    root = args.parent/'frozen-indexes'
    idx = SidIndex.from_full_sids(np.load(root/'items-42.npy'),width=256); wrapped=InputIndex4(idx)
    h,u,t = cases(args.data,'train')
    full = encode_window(h,idx.sids,32)
    pair = encode(h,idx.sids,np.load(root/'pair-keys.npy'),np.load(root/'pairs-42.npy'),32)
    views = MixedViews(full,pair)
    training_examples = (pair,u,t)
    exposure = []
    for epoch in range(1,61):
        order = np.random.default_rng(42+epoch).permutation(len(t))
        for b,begin in enumerate(range(0,len(t),512)):
            part = order[begin:begin+512]
            x = views(epoch,b,part)
            if x.shape[1] == 128:
                np.testing.assert_array_equal(x,full[part])
            else:
                np.testing.assert_array_equal(x,pair[part])
        exposure.append(dict(epoch=epoch,**views.epoch_summary(epoch)))
    model = attach4(build_grid_generator(idx.sids,c['generator'],42,'cpu',args.source))
    baseline = json.loads((args.parent/'trials/pair32_wide_42/trial-result.json').read_text())
    assert state_digest(dict(model.named_parameters())) == baseline['initial_state_sha256']
    h,u,t = cases(args.data,'validation')
    inputs = dict(pair32=encode(h[:2],idx.sids,np.load(root/'pair-keys.npy'),np.load(root/'pairs-42.npy'),32),
                  full32=encode_window(h[:2],idx.sids,32))
    before = state_digest(dict(model.named_parameters())); costs={}
    model.eval()
    for name,x in inputs.items():
        with torch.inference_mode(),torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH),torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],with_flops=True) as prof:
            loss = _loss(model,wrapped.model_batch(x[:1],u[:1],t[:1],'cpu'))
        assert torch.isfinite(loss)
        costs[name] = sum(e.flops for e in prof.key_averages())
    assert state_digest(dict(model.named_parameters())) == before
    assert costs['pair32'] == baseline['forward']['supported_forward_flops_batch1']
    optimizer = torch.optim.Adam(model.parameters(),lr=.003,weight_decay=.0001)
    losses = {}
    model.train()
    for name,x in inputs.items():
        optimizer.zero_grad(set_to_none=True)
        loss = _loss(model,wrapped.model_batch(x,u[:2],t[:2],'cpu')); loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        assert torch.isfinite(loss) and torch.isfinite(norm)
        losses[name] = float(loss.detach())
        optimizer.step()
    assert state_digest(dict(model.named_parameters())) != before
    report = dict(parent_files_verified=len(c['parent']['files']),input_files_verified=len(c['movie_input']['sha256']),
        train_targets=len(full),all_60_epoch_schedules_consume_every_example_once=True,
        full_examples=sum(e['examples']['full32'] for e in exposure),pair_examples=sum(e['examples']['pair32'] for e in exposure),
        native_encoder_positions=dict(full32=129,pair32=97),initial_parameters_match_completed_pair32=True,
        cpu_native_forward_backward_finite=losses,supported_forward_flops_batch1=costs,
        inference_flops_match_completed_pair32=True,policy='50/50 whole optimizer batches; independent RNG; Pair32-only evaluation',
        gpu_used=False)
    if c.get('resume_parent'):
        from rhpr.gr.generator_resume import restore_training
        from rhpr.gr.kindle_mixed_views import locate_resume
        from rhpr.gr.sid import evaluate_ranked_sids
        if args.resume is None:
            raise ValueError('--resume required for continuation preflight')
        resumed_root = locate_resume(c,args.resume)
        source = resumed_root/'trials/mixed32_wide_42'
        state = torch.load(source/'generator-last.pt',map_location='cpu',weights_only=False)
        trial_config = copy.deepcopy(c); trial_config['data']['history_items']=32
        validation_examples=(encode(h,idx.sids,np.load(root/'pair-keys.npy'),np.load(root/'pairs-42.npy'),32),u,t)
        with tempfile.TemporaryDirectory() as tmp:
            restored = restore_training(model,optimizer,torch.amp.GradScaler('cuda',enabled=False),wrapped,trial_config,
                training_examples,validation_examples,source,Path(tmp),'cpu')
            assert state_digest(model.state_dict())==state_digest(state['model'])
        assert restored['epoch']==30 and restored['updates']==8280 and restored['skipped']==0
        curves=json.loads((source/'training-curves.json').read_text())
        for row in curves:
            assert row['training_views']['schedule_sha256']==exposure[row['epoch']-1]['schedule_sha256']
        for row in curves:
            if 'validation' not in row: continue
            with np.load(source/f"validation-epoch-{row['epoch']:03d}.npz") as z:
                np.testing.assert_array_equal(z['user_ids'],u)
                np.testing.assert_array_equal(z['target_item_ids'],t)
                metrics=evaluate_ranked_sids(idx,z['semantic_ids'],t,cutoffs=(5,10))
                for name,value in metrics.items(): assert abs(value-row['validation'][name])<1e-12
        report.update(resume_files_verified=len(c['resume_parent']['files']),source_epoch=30,source_updates=8280,
            source_best_epoch=restored['best_epoch'],source_best_ndcg10=restored['best_metric'],
            actual_native_model_and_optimizer_restored=True,all_30_saved_view_schedules_match=True,
            six_validation_predictions_independently_replayed=True,
            resume_checkpoint_sha256=file_sha256(source/'generator-last.pt'))
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
