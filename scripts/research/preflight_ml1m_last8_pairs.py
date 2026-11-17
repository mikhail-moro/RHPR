"""CPU preflight of all Pair128 shapes with real item SIDs and synthetic pair SIDs."""
import json
import random
from functools import partial
from pathlib import Path

import numpy as np
import torch

from rhpr.gr.experiment import _loss
from rhpr.gr.grid import build_grid_generator
from rhpr.gr.ml100k_pair_raw import cases
from rhpr.gr.ml1m_last8_pairs import MERGE_PREFIXES,encode,cache_sets,pair_keys,validate,profiling_batch
from rhpr.gr.pair_unique4 import attach4,InputIndex4
from rhpr.gr.sid import SidIndex
from rhpr.gr.sid_study import state_digest
from rhpr.gr.movie_data import case_digest
from rhpr.gr.profiling import benchmark_training,ProfileDeadline

torch.set_num_threads(1)
c=json.loads(Path('experiments/ml1m_last8_pair128_2t4.json').read_text());validate(c)
data=Path('/tmp/ml1m-last8-stopped-v2/prepared-cases');source=Path('/workspace/artifacts/history-length-input/resources/grid-prepared')
raw={s:cases(data,s) for s in ('train','validation','test')}
assert {s:case_digest(a) for s,a in raw.items()}==c['expected_raw_case_sha256']
sets=cache_sets(data);keys=np.unique(np.concatenate(list(sets.values())))
assert {s:len(k) for s,k in sets.items()}==c['expected_pair_counts'] and len(keys)==c['expected_pair_union']
np.save('/tmp/ml1m-pair128-keys.npy',keys)
i=np.arange(len(keys));pairs=np.column_stack((i%256,(i//256)%256,(i//65536)%256,np.zeros(len(i)))).astype(np.int32)
sids=np.load('/tmp/ml1m-last8-stopped-v2/frozen-indexes/items-42.npy');index=InputIndex4(SidIndex.from_full_sids(sids,width=256))
random.seed(42);np.random.seed(42);torch.manual_seed(42)
model=build_grid_generator(sids,c['generator'],42,'cpu',source)
assert state_digest(dict(model.named_parameters()))==c['expected_initial_state_sha256']
before={k:v.clone() for k,v in model.named_parameters()};attach4(model)
assert all(torch.equal(v,dict(model.named_parameters())[k]) for k,v in before.items())
assert state_digest(dict(model.named_parameters()))==c['expected_pair_initial_state_sha256']
costs={};views={}
h,u,t=raw['train'];rows=np.argsort((h>=0).sum(1))[-2:]
for m in MERGE_PREFIXES:
    x=encode(h[rows],sids,keys,pairs,m);batch=index.model_batch(x,u[rows],t[rows],'cpu')
    model.eval()
    with torch.inference_mode(),torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH),torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU],with_flops=True) as prof:
        loss=_loss(model,{k:v[:1] for k,v in batch.items()})
    assert torch.isfinite(loss)
    model.train();model.zero_grad(set_to_none=True);loss=_loss(model,batch);loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(torch.nn.utils.clip_grad_norm_(model.parameters(),1.))
    assert model.pair_input_embedding.weight.grad is not None and model.pair_input_embedding.weight.grad.abs().sum()>0
    assert state_digest(dict(model.named_parameters()))==c['expected_pair_initial_state_sha256']
    costs[str(m)]=dict(protected_tail=128-m,encoder_positions=x.shape[1]+1,forward_flops=sum(e.flops for e in prof.key_averages()),finite_forward_backward=True)
    views[str(m)]={}
    for s,(rh,_,_) in raw.items():
        n=(rh[:,-128:]>=0).sum(1);p=np.maximum(n-(128-m),0)//2
        views[str(m)][s]=dict(pair_occurrences=int(p.sum()),examples_with_pairs=int((p>0).sum()),unique_pairs=len(pair_keys(rh,3706,m)),mean_valid_item_or_pair_units=float((n-p).mean()))
model.train();model.zero_grad(set_to_none=True)
compiled=torch.compile(model,backend='aot_eager',fullgraph=False,dynamic=False)
for i in rows:
    micro=index.model_batch(encode(h[i:i+1],sids,keys,pairs,64),u[i:i+1],t[i:i+1],'cpu')
    loss=_loss(compiled,micro);assert torch.isfinite(loss);(loss/len(rows)).backward()
assert torch.isfinite(torch.nn.utils.clip_grad_norm_(model.parameters(),1.))
assert state_digest(dict(model.named_parameters()))==c['expected_pair_initial_state_sha256']
probe=(encode(h[rows],sids,keys,pairs,64),u[rows],t[rows])
profile=benchmark_training(compiled,probe,sids,np.arange(2),1,'fp32','full_pipeline',
    {**c['training'],'batch_size':2},dict(seed=42,warmup=1,iterations=2),ProfileDeadline(180),'cpu',
    batch_builder=partial(profiling_batch,index))
assert profile['measured_updates_applied']==2 and profile['microbatches_per_update']==2
assert state_digest(dict(model.named_parameters()))==c['expected_pair_initial_state_sha256']
model.eval();batch=index.model_batch(encode(h[rows[:1]],sids,keys,pairs,64),u[rows[:1]],t[rows[:1]],'cpu')
with torch.inference_mode():
    paths,scores=model.generate(input_ids=batch['input_ids'],attention_mask=batch['attention_mask_encoder'],user_id=batch['user_id'])
assert paths.shape==(1,20,4) and torch.isfinite(scores).all()
r=dict(gpu_used=False,actual_item_index=True,synthetic_unique_pair_sids=True,pair_quantizer_not_fitted_locally=True,
    costs=costs,views=views,counts=c['expected_pair_counts'],pair_union=len(keys),pair_embedding_bytes=len(keys)*768*4,
    shared_base_initial_state_sha256=c['expected_initial_state_sha256'],pair_initial_state_sha256=c['expected_pair_initial_state_sha256'],
    parameters=sum(v.numel() for v in model.parameters()),added_input_embedding_parameters=131072,finite_beam20=True,
    common_base_parameters_identical=True,aot_eager_cpu_accumulation_finite=True,compiled_profile_accumulation_restored=True,
    cpu_profile=profile,gpu_compilation_not_tested_locally=True,
    raw_case_sha256=c['expected_raw_case_sha256'],reference_control_matched_best_epoch=35)
Path('docs/gr-ml1m-last8-pair128-preflight.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
