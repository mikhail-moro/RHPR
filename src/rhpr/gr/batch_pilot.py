"""Bounded T4 batch-throughput pilot using frozen legacy weights and real cases."""
from __future__ import annotations
import argparse, gc, hashlib, json, os, statistics, subprocess, sys, time, traceback
from pathlib import Path
from rhpr.gr.sdpa_pilot import new_model, write

BATCHES = [(256,128),(256,256),(512,512),(1024,1024)]


def validate(config):
    p=config['pilot']
    if (config['kind']!='gr_t5_batch_pilot' or p['history_items']!=[20,50]
            or p['backends']!=['legacy_eager','sdpa'] or p['batches']!=[list(x)for x in BATCHES]
            or p['warmup_steps']!=10 or p['timed_steps']!=50 or p['end_to_end_steps']!=10
            or p['max_seconds']!=1200 or p['new_transformers']!='5.22.1'
            or config['generator']['dropout']!=.1):
        raise ValueError('batch pilot differs from the bounded comparison contract')


def schedule(config):
    validate(config)
    return [dict(id=f'{backend}-h{width}-b{effective}-p{physical}',backend=backend,
                 history_items=width,batch_size=effective,physical_batch_size=physical)
            for width in config['pilot']['history_items']
            for effective,physical in BATCHES
            for backend in config['pilot']['backends']]


def slices(effective,physical):
    if effective<1 or physical<1 or physical>effective or effective%physical:
        raise ValueError('batch sizes must divide without changing the effective batch')
    return [slice(i,i+physical)for i in range(0,effective,physical)]


def training_cases(plan,width,effective):
    import numpy as np
    from rhpr.gr.movie_data import load_temporal_users,temporal_examples
    users=load_temporal_users(plan['sequences'],plan['source_ids'])
    values=temporal_examples(users,width,'train')
    # All configurations use prefixes of one fixed sample; paired backends get
    # byte-identical histories, users and targets. No heldout items are trained.
    chosen=np.random.default_rng(42).choice(len(values[2]),1024,replace=False)[:effective]
    return tuple(v[chosen]for v in values)


def adam_step(model,optimizer,batches,weights,scaler,device):
    """Same mean-loss accumulation as production, including uneven chunks."""
    import torch
    from rhpr.gr.experiment import _loss
    optimizer.zero_grad(set_to_none=True);losses=[]
    for batch,weight in zip(batches,weights,strict=True):
        with torch.autocast(device_type=torch.device(device).type,dtype=torch.float16,enabled=scaler.is_enabled()):
            loss=_loss(model,batch)
        if not torch.isfinite(loss):raise FloatingPointError('nonfinite FP16 batch-pilot loss')
        losses.append(loss.detach()*weight);scaler.scale(loss*weight).backward()
    scaler.unscale_(optimizer)
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
    if not torch.isfinite(norm):raise FloatingPointError('nonfinite batch-pilot gradient norm')
    before=scaler.get_scale();scaler.step(optimizer);scaler.update()
    if scaler.get_scale()<before:raise FloatingPointError('batch pilot skipped an AMP optimizer update')
    return sum(losses)


def attention_backends(model):
    # Transformers4 T5Attention has no .config; stack models do.
    return sorted({m.config._attn_implementation for m in model.modules()
                   if type(m).__name__ in {'T5Stack','T5EncoderModel'}})


def benchmark(plan,trial):
    import numpy as np,torch
    from rhpr.gr.experiment import _model_batch
    p=plan['config']['pilot'];backend=trial['backend'];effective=trial['batch_size'];physical=trial['physical_batch_size']
    model=new_model(plan,backend);model.train()
    data=training_cases(plan,trial['history_items'],effective);sids=np.load(plan['sids'])
    parts=slices(effective,physical);weights=[(s.stop-s.start)/effective for s in parts]
    build=lambda:[_model_batch(*(v[s]for v in data),sids,plan['device'])for s in parts]
    batches=build();optimizer=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=.0001)
    scaler=torch.amp.GradScaler('cuda',init_scale=1024)
    torch.manual_seed(123);torch.cuda.manual_seed_all(123)
    step=lambda:adam_step(model,optimizer,batches,weights,scaler,plan['device'])
    for _ in range(p['warmup_steps']):step()
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();curve=[]
    for _ in range(p['timed_steps']):
        start=time.perf_counter();loss=step();torch.cuda.synchronize()
        curve.append(dict(seconds=time.perf_counter()-start,loss=float(loss)))
    peak=torch.cuda.max_memory_allocated();reserved=torch.cuda.max_memory_reserved()
    # Release old batch tensors before rebuilding, avoiding artificially doubled
    # input storage at physical batch1024. Timed loop above uses resident inputs.
    batches.clear();end_to_end=[]
    for _ in range(p['end_to_end_steps']):
        start=time.perf_counter();batches=build();loss=step();torch.cuda.synchronize()
        end_to_end.append(dict(seconds=time.perf_counter()-start,loss=float(loss)));batches.clear()
    row=dict(**trial,status='complete',dropout=.1,train_precision='fp16',learning_rate=.001,
        warmup_steps=p['warmup_steps'],timed_steps=p['timed_steps'],end_to_end_steps=p['end_to_end_steps'],
        amp_skipped_updates=0,optimizer_updates=p['warmup_steps']+p['timed_steps']+p['end_to_end_steps'],
        examples_consumed=effective*(p['warmup_steps']+p['timed_steps']+p['end_to_end_steps']),
        mean_step_seconds=statistics.mean(x['seconds']for x in curve),
        median_step_seconds=statistics.median(x['seconds']for x in curve),
        examples_per_second=effective/statistics.mean(x['seconds']for x in curve),
        end_to_end_mean_seconds=statistics.mean(x['seconds']for x in end_to_end),
        end_to_end_examples_per_second=effective/statistics.mean(x['seconds']for x in end_to_end),
        peak_allocated_bytes=peak,peak_reserved_bytes=reserved,
        peak_allocated_including_e2e_bytes=torch.cuda.max_memory_allocated(),
        input_sha256=hashlib.sha256(b''.join(v.tobytes()for v in data)).hexdigest(),
        attention_backends=attention_backends(model),
        curve=curve,end_to_end_curve=end_to_end)
    if row['attention_backends']!=['sdpa'if backend=='sdpa'else'eager']:
        raise ValueError('actual T5 attention backend differs from requested configuration')
    return row


def child(plan_path,index):
    import torch,transformers
    torch.set_num_threads(1);plan=json.loads(Path(plan_path).read_text());trial=schedule(plan['config'])[index]
    plan['device']='cuda:0'
    if not torch.cuda.is_available()or torch.cuda.get_device_capability()!=(7,5):
        raise RuntimeError('batch pilot requires T4; no hardware fallback')
    expected='4.47.0'if trial['backend']=='legacy_eager'else'5.22.1'
    if transformers.__version__!=expected:raise ValueError('unexpected Transformers version')
    output=Path(plan['output']);started=time.monotonic()
    try:
        result=benchmark(plan,trial)
    except torch.cuda.OutOfMemoryError as exc:
        # Expected feasibility outcome, never lower/retry the requested batch.
        result=dict(**trial,status='oom',exception_type=type(exc).__name__,message=str(exc),
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
    except FloatingPointError as exc:
        result=dict(**trial,status='numerical_failed',exception_type=type(exc).__name__,message=str(exc))
    result.update(device=torch.cuda.get_device_name(),torch=torch.__version__,transformers=transformers.__version__,wall_seconds=time.monotonic()-started)
    write(output/(trial['id']+'.json'),result)
    print(json.dumps(dict(phase='batch_pilot_trial_complete',id=trial['id'],status=result['status'])),flush=True)


def run_batch_pilot(config,output):
    from rhpr.offline_runtime import resolve_dataset_directory
    from rhpr.gr.resources import verify_resources,unpack_grid,verify_parent_artifacts
    from rhpr.gr.grid import prepare_grid_source
    from rhpr.gr.movie_singleton import prepare_singleton_input
    from rhpr.gr.generator_suite import verify_file
    trials=schedule(config);output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    resources=resolve_dataset_directory(config['resources']['path']);verify_resources(resources,config['resources']['manifest_sha256'])
    data=resolve_dataset_directory(config['movie_input']['path']);verify_resources(data,config['movie_input']['manifest_sha256'])
    data=prepare_singleton_input(data,output/'singleton-input',config['data'])
    extra=resolve_dataset_directory(config['pilot_input']['path']);verify_resources(extra,config['pilot_input']['manifest_sha256'])
    parent=verify_parent_artifacts(config['parent'])
    for record in config['parent_trial_files']:verify_file(parent,record)
    trial_root=parent/'trials'/'sum_seed42';scratch=Path(config['resources']['scratch_path'])
    source=prepare_grid_source(unpack_grid(resources/'grid-source.bin',scratch/'original'),scratch/'grid')
    overlay=scratch/'modern-site';overlay.mkdir(parents=True,exist_ok=True)
    command=[sys.executable,'-m','pip','install','--no-index','--no-deps','--no-cache-dir','--no-compile','--require-hashes',
             '--only-binary=:all:','--target',str(overlay),'--find-links',str(extra),'-r',str(extra/'requirements-lock.txt')]
    with(output/'modern-install.log').open('w')as f:subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=180)
    plan=dict(config=config,source=str(source),sequences=str(data/'sequences.json'),
        source_ids=[r['item_id']for r in json.loads((data/'catalog-stage-01.json').read_text())],
        sids=str(trial_root/'semantic-ids.npy'),checkpoint=str(trial_root/'generator-best.pt'),output=str(output))
    write(output/'batch-plan.json',plan);started=time.monotonic();results=[]
    for i,trial in enumerate(trials):
        remaining=config['pilot']['max_seconds']-(time.monotonic()-started)
        if remaining<=0:raise TimeoutError('batch pilot total child budget exhausted')
        paths=[p for p in sys.path if p]
        if trial['backend']=='sdpa':paths.insert(0,str(overlay))
        env=dict(os.environ,PYTHONPATH=os.pathsep.join(paths),CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        cmd=[sys.executable,'-m','rhpr.gr.batch_pilot','--plan',str(output/'batch-plan.json'),'--trial',str(i)]
        print(json.dumps(dict(phase='batch_pilot_trial_start',**trial)),flush=True)
        with(output/(trial['id']+'.log')).open('w')as f:subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=remaining)
        results.append(json.loads((output/(trial['id']+'.json')).read_text()));write(output/'batch-results.json',results)
    for a,b in zip(results[::2],results[1::2],strict=True):
        if a['status']==b['status']=='complete' and a['input_sha256']!=b['input_sha256']:
            raise ValueError('paired backends used different training examples')
    successful=[r for r in results if r['status']=='complete']
    metrics=dict(experiment=config['name'],attempted=len(results),complete=len(successful),
        oom=sum(r['status']=='oom'for r in results),numerical_failed=sum(r['status']=='numerical_failed'for r in results),wall_seconds=time.monotonic()-started)
    for backend in config['pilot']['backends']:
        for width in config['pilot']['history_items']:
            good=[r for r in successful if r['backend']==backend and r['history_items']==width]
            if good:
                best=max(good,key=lambda r:r['examples_per_second']);prefix=f'{backend}_h{width}'
                metrics[prefix+'_best_examples_per_second']=best['examples_per_second']
                metrics[prefix+'_best_batch']=best['batch_size'];metrics[prefix+'_best_physical']=best['physical_batch_size']
    return metrics

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--trial',required=True,type=int)
    a=p.parse_args();child(a.plan,a.trial)
