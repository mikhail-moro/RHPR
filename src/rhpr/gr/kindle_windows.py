"""Matched Recent20/32/64 controls on one fixed long-reader Kindle cohort."""
from __future__ import annotations
import argparse, copy, gc, json, random, subprocess, sys, time, traceback
from pathlib import Path
import numpy as np
from rhpr.gr.experiment import _event, _loss, _train_generator, _write

SEEDS = [42, 101, 202]
WINDOWS = [20, 32, 64]
# Interleaved across two workers, balancing the longest fits first.
TRIALS = [dict(id=f'recent{w}_{s}', seed=s, history_items=w) for w,s in
          [(64,42),(64,101),(64,202),(32,42),(20,101),(32,101),(20,202),(32,202),(20,42)]]


def validate(c):
    t=c['training'];g=c['generator']
    if (c['kind']!='gr_kindle_windows' or c['seeds']!=SEEDS or c['windows']!=WINDOWS
        or c['trials']!=TRIALS or c['seed']!=42 or c['data']['history_items']!=64
        or c['evaluation']['test_after_selection'] is not False
        or (t['min_epochs'],t['max_epochs'],t['validation_interval'],t['patience_epochs'])!=(40,60,5,15)
        or (t['batch_size'],t['physical_batch_size'],t['evaluation_batch_size'])!=(512,512,64)
        or (t['learning_rate'],t['weight_decay'],t['grad_clip_norm'])!=(.003,.0001,1.)
        or t['precision']!='fp16' or t['evaluation_precision']!='fp32' or t['validate_first_epoch']
        or t['scheduler']!={'kind':'warmup_cosine','warmup_epochs':2,'total_epochs':60,'min_lr':.001}
        or t['early_stopping']!={'kind':'ema','alpha':.5,'min_delta':.0005}
        or t['compile_training']!={'mode':'reduce-overhead','fullgraph':False,'dynamic':False}
        or (g['encoder_layers'],g['decoder_layers'],g['d_model'],g['d_ff'])!=(4,4,64,512)
        or g['attn_implementation']!='sdpa' or c['execution']['worker_devices']!=['cuda:0','cuda:1']
        or c['quantizer']['num_hierarchies']!=3 or c['quantizer']['codebook_size']!=256):
        raise ValueError('Kindle window screen differs from declared paired protocol')


def encode_window(contexts, sids, width):
    """Keep the latest items, four codes/item, preserve left padding and order."""
    h=np.asarray(contexts)
    if width not in WINDOWS or h.ndim!=2 or h.shape[1]!=64 or sids.ndim!=2 or sids.shape[1]!=4:
        raise ValueError('expected 64-item histories and unique 3+1 SIDs')
    h=h[:,-width:]
    if np.any(h < -1) or np.any(h>=len(sids)):raise ValueError('invalid item ID')
    tokens=sids[np.maximum(h,0)]+np.arange(4,dtype=np.int32)*256
    return np.where(h[:,:,None]>=0,tokens,-1).reshape(len(h),4*width).astype(np.int32)


def run_trial(trial,p):
    import torch
    from rhpr.gr.grid import build_grid_generator
    from rhpr.gr.sid import SidIndex
    from rhpr.gr.sid_study import state_digest
    from rhpr.gr.pair_unique4 import attach4,InputIndex4
    from rhpr.gr.ml100k_pair_raw import cases
    from rhpr.gr.movie_data import case_digest
    from rhpr.gr.data import file_sha256
    c=copy.deepcopy(p['config']);seed=trial['seed'];width=trial['history_items'];c['seed']=seed;c['data']['history_items']=width
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    sidpath=Path(p['indexes'])/f'items-{seed}.npy';sid=np.load(sidpath,allow_pickle=False)
    idx=SidIndex.from_full_sids(sid,width=256);wrapped=InputIndex4(idx)
    model=attach4(build_grid_generator(idx.sids,c['generator'],seed,'cpu',p['source']))
    initial=state_digest(dict(model.named_parameters()))
    out=Path(p['output'])/'trials'/trial['id'];out.mkdir(parents=True,exist_ok=True)
    raw={s:cases(p['data'],s)for s in ('train','validation')}
    # CPU math attention exposes matmuls that fused CUDA SDPA hides from the counter.
    h,u,t=raw['validation'];probe=encode_window(h[:1],idx.sids,width)
    model.eval()
    with torch.inference_mode(),torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH),torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],with_flops=True)as prof:
        _loss(model,wrapped.model_batch(probe,u[:1],t[:1],'cpu'))
    flop_report=dict(parameters=sum(v.numel()for v in model.parameters()),encoder_positions=4*width+1,
        supported_forward_flops_batch1=sum(e.flops for e in prof.key_averages()),
        scope='B1 full validation context, CPU math SDPA, teacher-forced four-code forward with output heads; supported profiler operations, FMA=2; excludes backward/beam/text encoder')
    if state_digest(dict(model.named_parameters()))!=initial:raise RuntimeError('profiling changed initializer')
    _write(out/'forward-flops.json',flop_report)
    model.to('cuda:0');torch.cuda.reset_peak_memory_stats();started=time.monotonic()
    def builder(root,ignored_width,split):
        if split not in raw:raise ValueError('TEST is closed during window screening')
        h,u,t=raw[split];return encode_window(h,idx.sids,width),u,t
    selection=_train_generator(model,p['data'],wrapped,c,out,'cuda:0',example_builder=builder)
    epochs=selection['training_epochs'];expected=(len(raw['train'][2])+511)//512
    if not 40<=epochs<=60 or selection['validation_checks']!=epochs//5 or selection['stopping_reason']not in ('early_stopping','epoch_limit'):
        raise RuntimeError('trial did not complete its declared stopping protocol')
    if selection['optimizer_updates']+selection['amp_skipped_updates']!=expected*epochs:raise RuntimeError('update accounting mismatch')
    curves=json.loads((out/'training-curves.json').read_text());warm=[r['train_seconds']for r in curves[1:]]
    result=dict(trial=trial['id'],seed=seed,history_items=width,initial_state_sha256=initial,
        sid_file_sha256=file_sha256(sidpath),selection=selection,forward=flop_report,
        raw_case_sha256={s:case_digest(a)for s,a in raw.items()},
        case_targets_sha256={s:case_digest(a[1:])for s,a in raw.items()},
        timing=dict(wall_seconds=time.monotonic()-started,first_epoch_training_seconds=curves[0]['train_seconds'],
            warm_epoch_training_median_seconds=float(np.median(warm)),
            cuda_peak_allocated_bytes=int(torch.cuda.max_memory_allocated()),cuda_peak_reserved_bytes=int(torch.cuda.max_memory_reserved())),
        test_evaluated=False)
    _write(out/'trial-result.json',result);return result


def worker(plan_path,slot):
    import torch,transformers
    p=json.loads(Path(plan_path).read_text());torch.set_num_threads(1)
    if torch.cuda.device_count()!=1 or torch.cuda.get_device_capability()!=(7,5)or transformers.__version__!='5.22.1':raise RuntimeError('requires isolated T4/HF5.22.1')
    state=dict(results={},failures=[])
    for trial in p['assignments'][slot]['trials']:
        _event('kindle_windows_trial_start',trial=trial['id'],seed=trial['seed'],history_items=trial['history_items'])
        try:
            row=run_trial(trial,p);state['results'][trial['id']]=row;_event('kindle_windows_trial_complete',**row)
        except Exception as error:
            out=Path(p['output'])/'trials'/trial['id'];out.mkdir(parents=True,exist_ok=True);(out/'failure.log').write_text(traceback.format_exc())
            state['failures'].append(dict(trial=trial['id'],exception_type=type(error).__name__,message=str(error)))
            _event('kindle_windows_trial_failed',**state['failures'][-1])
        _write(Path(p['output'])/'workers'/f'worker-{slot}.json',state)
        torch._dynamo.reset();gc.collect();torch.cuda.empty_cache()


def run(c,output):
    import torch
    from sentence_transformers import SentenceTransformer
    from rhpr.gr.ae_suite import assignments,launch_workers
    from rhpr.gr.grid import fit_grid_rkmeans,prepare_grid_source
    from rhpr.gr.data import file_sha256
    from rhpr.gr.resources import unpack_grid,verify_resources
    from rhpr.gr.sid import SidIndex
    from rhpr.offline_runtime import resolve_dataset_directory
    validate(c);output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True);(output/'workers').mkdir(exist_ok=True)
    if torch.cuda.device_count()!=2 or any(torch.cuda.get_device_capability(i)!=(7,5)for i in (0,1)):raise RuntimeError('requires two T4s; no hardware fallback')
    torch.set_num_threads(1);data=resolve_dataset_directory(c['movie_input']['path'])
    for name,digest in c['movie_input']['sha256'].items():
        if file_sha256(data/name)!=digest:raise ValueError('input digest mismatch: '+name)
    report=json.loads((data/'data-report.json').read_text())
    if any(report[k]!=c['data'][k]for k in ('users','items','train_examples')):raise ValueError('unexpected cohort')
    resources=resolve_dataset_directory(c['resources']['path']);verify_resources(resources,c['resources']['manifest_sha256'])
    modern=resolve_dataset_directory(c['modern_runtime']['path']);verify_resources(modern,c['modern_runtime']['manifest_sha256'])
    scratch=Path(c['resources']['scratch_path']);scratch.mkdir(parents=True,exist_ok=True)
    source=prepare_grid_source(unpack_grid(resources/'grid-source.bin',scratch/'original'),scratch/'grid')
    cache=output/'item-cache';cache.mkdir();indexes=output/'frozen-indexes';indexes.mkdir()
    texts=json.loads((data/'texts.json').read_text());started=time.monotonic()
    encoder=SentenceTransformer(str(resources/'sentence-t5-base'),device='cuda:0',local_files_only=True);encoder.max_seq_length=2048;encoder.eval()
    vec=np.asarray(encoder.encode(texts,batch_size=32,convert_to_numpy=True,show_progress_bar=False),dtype=np.float32)
    if vec.shape[0]!=c['data']['items']or not np.isfinite(vec).all():raise ValueError('invalid embeddings')
    np.save(cache/'item-embeddings.npy',vec);del encoder;gc.collect();torch.cuda.empty_cache()
    _event('kindle_windows_embeddings_ready',items=len(vec),seconds=time.monotonic()-started)
    for seed in SEEDS:
        _event('kindle_windows_index_start',seed=seed)
        codes,centers,diagnostics=fit_grid_rkmeans(vec,c['quantizer'],seed,'cuda:0',source);idx=SidIndex(codes,width=256)
        np.save(indexes/f'items-{seed}.npy',idx.sids);np.save(indexes/f'item-centroids-{seed}.npy',centers)
        _write(indexes/f'index-{seed}.json',dict(fit=diagnostics,sid=idx.diagnostics(),max_suffix=int(idx.sids[:,-1].max())))
        _event('kindle_windows_index_complete',seed=seed)
    del vec,codes,centers,idx;gc.collect();torch.cuda.empty_cache()
    overlay=scratch/'modern-site';overlay.mkdir(parents=True,exist_ok=True)
    command=[sys.executable,'-m','pip','install','--no-index','--no-deps','--no-cache-dir','--no-compile','--require-hashes','--only-binary=:all:','--target',str(overlay),'--find-links',str(modern),'-r',str(modern/'requirements-lock.txt')]
    with (output/'modern-install.log').open('w')as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)
    jobs=assignments(c,2);plan=dict(config=c,output=str(output),source=str(source),data=str(data),indexes=str(indexes),assignments=jobs)
    _write(output/'plan.json',plan);sys.path.insert(0,str(overlay))
    summary=launch_workers(output/'plan.json',worker_module='rhpr.gr.kindle_windows',event_prefix='kindle_windows')
    rows=summary['results']
    for seed in SEEDS:
        matched=[rows[f'recent{w}_{seed}']for w in WINDOWS]
        for key in ('initial_state_sha256','sid_file_sha256','raw_case_sha256','case_targets_sha256'):
            if len({json.dumps(r[key],sort_keys=True)for r in matched})!=1:raise RuntimeError('unmatched '+key)
    _write(output/'study-summary.json',dict(results=rows,data=report,seed_scope='paired tokenizer+generator seeds; not a crossed variance decomposition',test_evaluated=False))
    return dict(experiment=c['name'],models=len(rows),users=report['users'],items=report['items'],test_evaluated=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--slot',type=int,required=True);a=p.parse_args();worker(a.plan,a.slot)
