"""CPU-only preflight on pinned real inputs and the native GRID generator."""
import argparse
import json
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
    args = parser.parse_args()
    c = json.loads(Path('experiments/kindle_mixed_views.json').read_text()); validate(c)
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
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
