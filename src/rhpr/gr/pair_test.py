"""Read-only test of fixed pair-SID and Recent10 generator checkpoints."""
from __future__ import annotations
import argparse, gc, json, random, subprocess, sys, traceback
from pathlib import Path
import numpy as np
from rhpr.gr.experiment import _event, _evaluate, _write
from rhpr.gr.ml1m_clean_mtl import case_examples
from rhpr.gr.movie_data import case_digest
from rhpr.gr.pair_sid import InputIndex, attach_pair_encoder, collect_pairs, encode_pairs, preencode, predict_rq
from rhpr.gr.pair_recent10 import recent_preencode

TRIALS = [dict(id=f'{mode}_{seed}_{which}', mode=mode, seed=seed, checkpoint=which)
          for mode in ('control', 'pair', 'recent10') for seed in (42, 101, 202) for which in ('last', 'best')]


def validate(c):
    if (c['kind'] != 'gr_ml1m_pair_test' or c['trials'] != TRIALS or c['seeds'] != [42, 101, 202]
        or c['data']['history_items'] != 20 or c['training']['evaluation_precision'] != 'fp32'
        or c['training']['evaluation_batch_size'] != 32 or c['selection_policy'] != 'validation_ndcg10_or_fixed_epoch50'
        or c['test_training_forbidden'] is not True or c['generator']['attn_implementation'] != 'sdpa'):
        raise ValueError('test protocol differs from predeclared evaluation')


def locate_parent(spec, input_root=Path('/kaggle/input')):
    from rhpr.gr.data import file_sha256
    owner, slug = spec['kernel'].split('/')
    candidates = [input_root / slug, input_root / owner / slug, input_root / 'notebooks' / owner / slug,
                  input_root / 'kernels' / owner / slug]
    if input_root.exists():
        candidates.extend(p.parent for p in input_root.rglob('run-manifest.json') if slug in p.parts)
    for root in candidates:
        if not all((root / f).is_file() for f in spec['files']): continue
        for name, expected in spec['files'].items():
            path = root / name
            if path.stat().st_size != expected['size_bytes'] or file_sha256(path) != expected['sha256']:
                raise ValueError(f'parent artifact differs: {name}')
        if json.loads((root / 'run-manifest.json').read_text())['git_sha'] != spec['git_sha']:
            raise ValueError('parent revision differs')
        return root
    raise FileNotFoundError(f"pinned parent not mounted: {spec['kernel']}")


def cache_plan(old_keys, contexts):
    """Keep original cached codes byte-for-byte; only encode unseen test pairs."""
    old_keys = np.asarray(old_keys)
    if old_keys.ndim != 1 or np.any(old_keys[1:] <= old_keys[:-1]): raise ValueError('old keys must be sorted and unique')
    needed = np.array(sorted(collect_pairs(contexts)), dtype=np.int64)
    missing = np.setdiff1d(needed, old_keys, assume_unique=True)
    return needed, missing, np.union1d(old_keys, missing)


def prepare_cache(plan_path):
    import torch
    from sentence_transformers import SentenceTransformer
    p = json.loads(Path(plan_path).read_text()); c = p['config']; validate(c)
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_capability() != (7, 5): raise RuntimeError('requires a T4')
    torch.set_num_threads(1); parent = Path(p['parents']['pair']); output = Path(p['cache']); output.mkdir(parents=True, exist_ok=True)
    old = np.load(parent / 'frozen-indexes/pair-keys.npy'); contexts, _, _ = case_examples(p['data_root'], 20, 'test')
    needed, missing, keys = cache_plan(old, contexts); np.save(output / 'pair-keys.npy', keys)
    _event('pair_test_cache_start', test_pairs=len(needed), missing_pairs=len(missing), reused_pairs=len(needed)-len(missing))
    if len(missing):
        texts = json.loads((Path(p['data_root']) / 'metadata-texts.json').read_text())
        encoder = SentenceTransformer(str(Path(p['resources']) / 'sentence-t5-base'), device='cuda:0', local_files_only=True)
        encoder.max_seq_length = 2048; encoder.eval(); encode_pairs(encoder, texts, missing, output / 'new-embeddings.npy')
        del encoder; gc.collect(); torch.cuda.empty_cache(); embeddings = np.load(output / 'new-embeddings.npy', mmap_mode='r')
    for seed in (42, 101, 202):
        oldcodes = np.load(parent / 'frozen-indexes' / f'pairs-{seed}.npy')
        if oldcodes.shape != (len(old), 3): raise ValueError('saved codes and keys differ')
        result = np.empty((len(keys),3),dtype=oldcodes.dtype); result[np.searchsorted(keys, old)] = oldcodes
        if len(missing):
            source = Path(p['centroids']) if seed != 202 else parent / 'frozen-indexes'
            centers = np.load(source / f'pair-centroids-{seed}.npy'); result[np.searchsorted(keys, missing)] = predict_rq(embeddings, centers, 'cuda:0')
        if not np.array_equal(result[np.searchsorted(keys, old)], oldcodes): raise RuntimeError('cached codes changed')
        np.save(output / f'pairs-{seed}.npy', result)
    if len(missing):
        del embeddings; (output / 'new-embeddings.npy').unlink()
    _write(output / 'cache-report.json', dict(test_unique_pairs=len(needed),missing_test_pairs=len(missing),
        reused_test_pairs=len(needed)-len(missing),quantizer_refit=False,cache_reuses_previous_assignments=True))
    _event('pair_test_cache_complete', test_pairs=len(needed), missing_pairs=len(missing))


def replay_trial(trial, p):
    import torch
    from rhpr.gr.data import file_sha256
    from rhpr.gr.grid import build_grid_generator
    from rhpr.gr.sid import SidIndex
    from rhpr.gr.sid_study import state_digest
    c=p['config']; mode=trial['mode']; seed=trial['seed']; which=trial['checkpoint']; ident=f'{mode}_{seed}'
    pair_parent=Path(p['parents']['pair']); parent=Path(p['parents']['recent'] if mode=='recent10' else p['parents']['pair'])
    old=json.loads((parent/'study-summary.json').read_text())['results'][ident]; expected_epoch=50 if which=='last' else old['selection']['best_epoch']
    ckpath=parent/'trials'/ident/f'generator-{which}.pt'; ck=torch.load(ckpath,map_location='cpu',weights_only=False)
    if ck['epoch']!=expected_epoch or ck['config']['seed']!=seed or ck['config']['generator']!=c['generator'] or ck['config']['data']!=c['data']:
        raise ValueError('checkpoint provenance or selected epoch mismatch')
    idx=SidIndex.from_full_sids(np.load(pair_parent/'frozen-indexes'/f'items-{seed}.npy'),width=256)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    model=attach_pair_encoder(build_grid_generator(idx.sids,c['generator'],seed,'cuda:0',p['source']))
    model.load_state_dict(ck['model'],strict=True); del ck; model.eval(); before=state_digest(model.state_dict())
    if mode=='pair':
        keys=np.load(Path(p['cache'])/'pair-keys.npy'); codes=np.load(Path(p['cache'])/f'pairs-{seed}.npy')
    else: keys=codes=None
    directory=Path(p['output'])/'trials'/trial['id']; directory.mkdir(parents=True,exist_ok=True); results={}
    for split in ('validation','test'):
        h,u,t=case_examples(p['data_root'],20,split)
        x=recent_preencode(h,idx.sids) if mode=='recent10' else preencode(h,idx.sids,keys,codes)
        score,predictions,probabilities=_evaluate(model,(x,u,t),InputIndex(idx),32,'cuda:0','fp32')
        if split=='validation':
            expected=old['epochs'][str(expected_epoch)]; error=score['ndcg@10']-expected
            if abs(error)>0.00005: raise RuntimeError(f'validation replay mismatch: {error}')
        np.savez_compressed(directory/f'{split}-predictions.npz',semantic_ids=predictions,scores=probabilities,user_ids=u,target_item_ids=t)
        results[split]=score
    if state_digest(model.state_dict())!=before: raise RuntimeError('inference modified model state')
    result=dict(trial=trial['id'],mode=mode,seed=seed,checkpoint=which,epoch=expected_epoch,
        checkpoint_sha256=file_sha256(ckpath),validation_expected_ndcg10=old['epochs'][str(expected_epoch)],
        validation=results['validation'],test=results['test'],training_performed=False)
    _write(directory/'result.json',result); return result


def worker(plan_path,slot):
    import torch,transformers
    p=json.loads(Path(plan_path).read_text());validate(p['config'])
    if torch.cuda.device_count()!=1 or torch.cuda.get_device_capability()!=(7,5) or transformers.__version__!='5.22.1': raise RuntimeError('worker needs T4 and HF 5.22.1')
    torch.set_num_threads(1); status=dict(results={},failures=[]); path=Path(p['output'])/'workers'/f'worker-{slot}.json'
    for trial in p['assignments'][slot]['trials']:
        _event('pair_test_trial_start',trial=trial['id'],worker=slot)
        try:
            result=replay_trial(trial,p);status['results'][trial['id']]=result;_event('pair_test_trial_complete',**result)
        except Exception as e:
            folder=Path(p['output'])/'trials'/trial['id'];folder.mkdir(parents=True,exist_ok=True);(folder/'failure.log').write_text(traceback.format_exc())
            status['failures'].append(dict(trial=trial['id'],exception_type=type(e).__name__,message=str(e)))
            _event('pair_test_trial_failed',trial=trial['id'],exception_type=type(e).__name__,message=str(e))
        _write(path,status);gc.collect();torch.cuda.empty_cache()


def run(c,output):
    import torch
    from rhpr.gr.ae_suite import assignments,launch_workers
    from rhpr.gr.grid import prepare_grid_source
    from rhpr.gr.resources import unpack_grid,verify_resources
    from rhpr.gr.study_parallel import worker_environment
    from rhpr.offline_runtime import resolve_dataset_directory
    validate(c);output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True);(output/'workers').mkdir(exist_ok=True)
    if torch.cuda.device_count()!=2 or any(torch.cuda.get_device_capability(i)!=(7,5) for i in (0,1)): raise RuntimeError('two T4 GPUs required')
    torch.set_num_threads(1);roots={}
    for name in ('resources','movie_input','modern_runtime','centroids'):
        roots[name]=resolve_dataset_directory(c[name]['path']);verify_resources(roots[name],c[name]['manifest_sha256'])
    from rhpr.gr.data import file_sha256
    if file_sha256(roots['movie_input']/'metadata-texts.json') != c['metadata_texts_sha256']: raise ValueError('metadata text changed')
    for split in ('validation','test'):
        if case_digest(case_examples(roots['movie_input'],20,split))!=c['data']['case_sha256'][split]:raise ValueError('case split differs')
    parents={k:locate_parent(v) for k,v in c['parents'].items()}
    for key in parents:
        old=json.loads((parents[key]/'run-manifest.json').read_text())['config']
        for name in ('training','generator','data'):
            if c[name]!=old[name]:raise ValueError(f'{key} parent {name} changed')
    scratch=Path(c['resources']['scratch_path']);scratch.mkdir(parents=True,exist_ok=True)
    source=prepare_grid_source(unpack_grid(roots['resources']/'grid-source.bin',scratch/'original'),scratch/'grid')
    plan=dict(config=c,output=str(output),source=str(source),data_root=str(roots['movie_input']),resources=str(roots['resources']),
        centroids=str(roots['centroids']),parents={k:str(v) for k,v in parents.items()},cache=str(output/'pair-cache'),assignments=assignments(c,2))
    _write(output/'plan.json',plan)
    env=worker_environment(c,plan['assignments'][0]); subprocess.run([sys.executable,'-u','-m','rhpr.gr.pair_test','--prepare-cache',str(output/'plan.json')],env=env,check=True,timeout=1800)
    overlay=scratch/'modern-site';overlay.mkdir(parents=True,exist_ok=True)
    command=[sys.executable,'-m','pip','install','--no-index','--no-deps','--no-cache-dir','--no-compile','--require-hashes','--only-binary=:all:',
        '--target',str(overlay),'--find-links',str(roots['modern_runtime']),'-r',str(roots['modern_runtime']/'requirements-lock.txt')]
    with (output/'modern-install.log').open('w') as f:subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=180)
    sys.path.insert(0,str(overlay));gc.collect();torch.cuda.empty_cache(); _event('pair_test_suite_start',trials=len(c['trials']))
    summary=launch_workers(output/'plan.json',worker_module='rhpr.gr.pair_test',event_prefix='pair_test')
    _write(output/'study-summary.json',dict(results=summary['results'],test_evaluated=True,training_performed=False))
    return dict(experiment=c['name'],models=9,checkpoints=len(summary['results']),test_evaluated=True,training_performed=False)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--plan');ap.add_argument('--slot',type=int);ap.add_argument('--prepare-cache');a=ap.parse_args()
    if a.prepare_cache:prepare_cache(a.prepare_cache)
    elif a.plan and a.slot is not None:worker(a.plan,a.slot)
    else:ap.error('provide --prepare-cache or --plan and --slot')
