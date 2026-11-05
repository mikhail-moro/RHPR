"""Bounded native-T5 SDPA pilot. Historical and modern runtimes stay isolated."""
from __future__ import annotations
import argparse, contextlib, copy, gc, hashlib, json, os, statistics, subprocess, sys, time
from pathlib import Path


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate(config):
    p = config['pilot']
    if (config['kind'] != 'gr_sdpa_pilot' or p['history_items'] != [20, 50]
            or p['timed_steps'] != 50 or p['warmup_steps'] != 10
            or p['batch_size'] != 256 or p['physical_batch_size'] != 128
            or p['max_seconds'] != 1200 or p['new_transformers'] != '5.22.1'
            or config['generator'].get('dropout') != 0.1):
        raise ValueError('pilot must keep its bounded comparison contract')


def compare_tensors(actual, reference, *, atol, rtol):
    import torch
    if set(actual) != set(reference):
        raise ValueError('tensor names differ between implementations')
    max_abs, sum_delta, sum_ref = 0., 0., 0.
    for name in sorted(actual):
        x, y = actual[name].float(), reference[name].float()
        if not torch.isfinite(x).all() or not torch.isfinite(y).all():
            raise FloatingPointError(f'nonfinite parity tensor: {name}')
        if x.shape != y.shape:
            raise ValueError(f'parity shape changed: {name}')
        diff = x - y
        max_abs = max(max_abs, float(diff.abs().max()))
        sum_delta += float(diff.square().sum()); sum_ref += float(y.square().sum())
        torch.testing.assert_close(x, y, atol=atol, rtol=rtol, msg=lambda m: name + ': ' + m)
    return dict(max_absolute_error=max_abs, relative_l2=(sum_delta / max(sum_ref, 1e-30))**.5,
                tensors=len(actual), atol=atol, rtol=rtol)


def examples(plan, width, split):
    import numpy as np
    from rhpr.gr.movie_data import load_temporal_users, temporal_examples
    users = load_temporal_users(plan['sequences'], plan['source_ids'])
    values = temporal_examples(users, width, split)
    if split == 'train':
        chosen = np.random.default_rng(42).choice(len(values[2]), 256, replace=False)
    else:
        chosen = np.linspace(0, len(values[2])-1, 32, dtype=np.int64)
    return tuple(v[chosen] for v in values)


def new_model(plan, backend, *, tiny=False):
    import numpy as np, torch
    from rhpr.gr.grid import build_grid_generator
    cfg = copy.deepcopy(plan['config']['generator'])
    if backend != 'legacy_eager': cfg['attn_implementation'] = backend
    if tiny: cfg.update(encoder_layers=1, decoder_layers=1, d_model=16, d_ff=32, num_heads=2, d_kv=8)
    model = build_grid_generator(np.load(plan['sids']), cfg, 42, plan['device'], plan['source'])
    state = torch.load(plan['checkpoint'], map_location='cpu', weights_only=False)['model']
    model.load_state_dict(state, strict=True)
    return model


def parity(plan, backend):
    import numpy as np, torch
    from rhpr.gr.experiment import _model_batch, _loss
    model = new_model(plan, backend)
    model.eval()  # dropout disabled for numerical comparison; gradients enabled
    sids = np.load(plan['sids']); data = examples(plan, 20, 'validation')
    batch = _model_batch(*(v[:8] for v in data), sids, plan['device'])
    hidden = model(**batch); loss = _loss(model, batch); loss.backward()
    tensors = {'hidden': hidden.detach().cpu(), 'loss': loss.detach().cpu()}
    gradients = {n: p.grad.detach().cpu() for n,p in model.named_parameters() if p.grad is not None}
    if not any('relative_attention_bias' in n for n in gradients):
        raise ValueError('relative-position bias gradients missing')
    with torch.inference_mode():
        paths, probs = model.generate(attention_mask=batch['attention_mask_encoder'], input_ids=batch['input_ids'], user_id=batch['user_id'])
    result = dict(loss=float(loss.detach()), gradient_tensors=len(gradients),
                  parameters=sum(p.numel() for p in model.parameters()),
                  prediction_shape=list(paths.shape))
    root = Path(plan['output'])
    torch.save(dict(tensors=tensors, gradients=gradients, paths=paths.cpu(), probs=probs.cpu()), root / (backend+'-parity.pt'))
    if backend != 'legacy_eager':
        old = torch.load(root/'legacy_eager-parity.pt', weights_only=False)
        result['legacy_output_parity'] = compare_tensors(tensors, old['tensors'], atol=2e-4, rtol=2e-4)
        result['legacy_gradient_parity'] = compare_tensors(gradients, old['gradients'], atol=3e-4, rtol=3e-3)
        result['legacy_beam_exact_fraction'] = float((paths.cpu() == old['paths']).all(-1).float().mean())
        result['legacy_beam_scores'] = compare_tensors({'scores':probs.cpu()}, {'scores':old['probs']}, atol=2e-6, rtol=2e-3)
        if result['legacy_beam_exact_fraction'] != 1.:
            raise ValueError('cached beam search differs from legacy on fixed parity examples')
    if backend == 'sdpa':
        eager = torch.load(root/'eager-parity.pt', weights_only=False)
        result['modern_output_parity'] = compare_tensors(tensors, eager['tensors'], atol=2e-4, rtol=2e-4)
        result['modern_gradient_parity'] = compare_tensors(gradients, eager['gradients'], atol=3e-4, rtol=3e-3)
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return result


def benchmark(plan, backend, width):
    import numpy as np, torch
    from rhpr.gr.experiment import _model_batch, _loss
    cfg = plan['config']['pilot']; model = new_model(plan, backend); model.train()
    data = examples(plan, width, 'train'); sids = np.load(plan['sids'])
    cpu_batches = [_model_batch(*(v[i:i+128] for v in data), sids, 'cpu') for i in (0,128)]
    batches = [{k:v.to(plan['device']) for k,v in b.items()} for b in cpu_batches]
    optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=.0001)
    scaler = torch.amp.GradScaler('cuda', init_scale=1024)
    torch.manual_seed(123); torch.cuda.manual_seed_all(123)
    history=[]; skipped=0
    def step():
        nonlocal skipped
        optimizer.zero_grad(set_to_none=True); losses=[]
        for batch in batches:
            with torch.autocast('cuda', dtype=torch.float16): loss = _loss(model, batch)
            if not torch.isfinite(loss): raise FloatingPointError('nonfinite FP16 pilot loss')
            losses.append(loss.detach()); scaler.scale(loss*.5).backward()
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        if not torch.isfinite(norm): skipped+=1
        scaler.step(optimizer); scaler.update()
        return torch.stack(losses).mean()
    for _ in range(cfg['warmup_steps']): step()
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    for _ in range(cfg['timed_steps']):
        started=time.perf_counter(); loss=step(); torch.cuda.synchronize()
        history.append(dict(seconds=time.perf_counter()-started,loss=float(loss)))
    peak_allocated=torch.cuda.max_memory_allocated(); peak_reserved=torch.cuda.max_memory_reserved()
    # End-to-end includes the production NumPy/SID batch preparation + H2D.
    e2e=[]
    for _ in range(10):
        started=time.perf_counter()
        batches=[_model_batch(*(v[i:i+128] for v in data),sids,plan['device'])for i in (0,128)]
        step();torch.cuda.synchronize();e2e.append(time.perf_counter()-started)
    # Profiler is outside measured steps. Preserve actual dispatched kernels.
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
        step();torch.cuda.synchronize()
    events=prof.key_averages();ops=[dict(name=e.key,count=e.count,cpu_time_total=e.cpu_time_total,device_time_total=getattr(e,'device_time_total',0.))for e in events]
    ops.sort(key=lambda r:r['device_time_total'],reverse=True)
    Path(plan['output'],f'{backend}-{width}-profile.txt').write_text(prof.key_averages().table(sort_by='cuda_time_total',row_limit=50))
    sdpa_ops=[r for r in ops if any(s in r['name'].lower()for s in ['scaled_dot','efficient_attention','flash_attention','cudnn_attention'])]
    if backend=='sdpa' and not sdpa_ops: raise ValueError('no SDPA operators found in runtime profile')
    result=dict(backend=backend,history_items=width,dropout=.1,train_precision='fp16',batch_size=256,physical_batch_size=128,
        warmup_steps=cfg['warmup_steps'],timed_steps=cfg['timed_steps'],amp_skipped_updates=skipped,
        mean_step_seconds=statistics.mean(r['seconds']for r in history),median_step_seconds=statistics.median(r['seconds']for r in history),
        end_to_end_mean_seconds=statistics.mean(e2e),peak_allocated_bytes=peak_allocated,peak_reserved_bytes=peak_reserved,
        loss_curve=history,profile_operators=ops[:50],sdpa_operators=sdpa_ops)
    if skipped: raise FloatingPointError('pilot had skipped AMP updates')
    del model,optimizer,batches,cpu_batches;gc.collect();torch.cuda.empty_cache()
    return result


def child(plan_path, backend, cpu=False):
    import torch,transformers
    torch.set_num_threads(1)
    plan=json.loads(Path(plan_path).read_text());plan['device']='cpu'if cpu else'cuda:0'
    if not cpu and (not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7,5)):
        raise RuntimeError('pilot requires a T4; no hardware fallback')
    expected='4.47.0'if backend=='legacy_eager'else'5.22.1'
    if transformers.__version__!=expected:raise ValueError('child runtime version mismatch')
    result=dict(backend=backend,transformers=transformers.__version__,torch=torch.__version__,
        device='cpu'if cpu else torch.cuda.get_device_name(),parity=parity(plan,backend),benchmarks=[])
    write(Path(plan['output'])/(backend+'-result.json'),result)
    if not cpu:
        for width in plan['config']['pilot']['history_items']:
            result['benchmarks'].append(benchmark(plan,backend,width))
            write(Path(plan['output'])/(backend+'-result.json'),result)
    print(json.dumps(dict(phase='sdpa_pilot_child_complete',backend=backend)),flush=True)


def run_pilot(config, output):
    from rhpr.offline_runtime import resolve_dataset_directory
    from rhpr.gr.resources import verify_resources,unpack_grid,verify_parent_artifacts
    from rhpr.gr.grid import prepare_grid_source
    from rhpr.gr.movie_singleton import prepare_singleton_input
    validate(config);output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    resources=resolve_dataset_directory(config['resources']['path']);verify_resources(resources,config['resources']['manifest_sha256'])
    data=resolve_dataset_directory(config['movie_input']['path']);verify_resources(data,config['movie_input']['manifest_sha256'])
    data=prepare_singleton_input(data,output/'singleton-input',config['data'])
    extra=resolve_dataset_directory(config['pilot_input']['path']);verify_resources(extra,config['pilot_input']['manifest_sha256'])
    parent=verify_parent_artifacts(config['parent'])
    from rhpr.gr.generator_suite import verify_file
    for record in config['parent_trial_files']:
        verify_file(parent,record)
    trial=parent/'trials'/'sum_seed42'
    scratch=Path(config['resources']['scratch_path']);source=prepare_grid_source(unpack_grid(resources/'grid-source.bin',scratch/'original'),scratch/'grid')
    overlay=scratch/'modern-site';overlay.mkdir(parents=True,exist_ok=True)
    command=[sys.executable,'-m','pip','install','--no-index','--no-deps','--no-cache-dir','--no-compile','--require-hashes',
        '--only-binary=:all:','--target',str(overlay),'--find-links',str(extra),'-r',str(extra/'requirements-lock.txt')]
    with (output/'modern-install.log').open('w')as f:subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=180)
    plan=dict(config=config,source=str(source),sequences=str(data/'sequences.json'),source_ids=[r['item_id']for r in json.loads((data/'catalog-stage-01.json').read_text())],
        sids=str(trial/'semantic-ids.npy'),checkpoint=str(trial/'generator-best.pt'),output=str(output))
    write(output/'pilot-plan.json',plan)
    started=time.monotonic();results={}
    # Sequential on the same physical T4: runtime and backend changes are separated.
    for backend in ['legacy_eager','eager','sdpa']:
        remaining=config['pilot']['max_seconds']-(time.monotonic()-started)
        if remaining<=0:raise TimeoutError('pilot time budget exhausted')
        paths=[p for p in sys.path if p]
        if backend!='legacy_eager':paths.insert(0,str(overlay))
        env=dict(os.environ,PYTHONPATH=os.pathsep.join(paths),CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
        cmd=[sys.executable,'-m','rhpr.gr.sdpa_pilot','--plan',str(output/'pilot-plan.json'),'--backend',backend]
        print(json.dumps(dict(phase='sdpa_pilot_child_start',backend=backend)),flush=True)
        with (output/(backend+'.log')).open('w')as f:subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=remaining)
        results[backend]=json.loads((output/(backend+'-result.json')).read_text())
    write(output/'pilot-results.json',results)
    metrics=dict(experiment=config['name'],parity_passed=True,backends=3,wall_seconds=time.monotonic()-started)
    for width in config['pilot']['history_items']:
        rows={b:next(r for r in v['benchmarks']if r['history_items']==width)for b,v in results.items()}
        for base in ['legacy_eager','eager']:
            metrics[f'sdpa_speedup_vs_{base}_h{width}']=rows[base]['mean_step_seconds']/rows['sdpa']['mean_step_seconds']
        metrics[f'sdpa_step_seconds_h{width}']=rows['sdpa']['mean_step_seconds']
        metrics[f'sdpa_memory_bytes_h{width}']=rows['sdpa']['peak_allocated_bytes']
    return metrics


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--backend',choices=['legacy_eager','eager','sdpa'],required=True);p.add_argument('--cpu',action='store_true')
    a=p.parse_args();child(a.plan,a.backend,a.cpu)
