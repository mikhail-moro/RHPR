"""Fixed-prefix Pair128 screens with optional full/pair training views."""
from __future__ import annotations

import argparse
import copy
import gc
import json
import random
import re
import subprocess
import sys
import time
import traceback
from functools import partial
from pathlib import Path

import numpy as np

from rhpr.gr.data import file_sha256
from rhpr.gr.experiment import _evaluate, _event, _loss, _train_generator, _write
from rhpr.gr.ml1m_last8 import REFERENCE_SID_SHA256, prepare_raw
from rhpr.gr.movie_data import case_digest

MERGE_PREFIXES = [64, 96, 112, 120]
TRIALS = [dict(id=f'pair128_merge{m}_42', merge_items=m, seed=42) for m in MERGE_PREFIXES]
MIXED_TRIALS = [dict(id=f'mixed128_merge{m}_42', merge_items=m, seed=42) for m in MERGE_PREFIXES]
REPLICATION_SEEDS = [101, 202]
REPLICATION_TRIALS = [dict(id=f'mixed128_merge120_{seed}', merge_items=120, seed=seed)
                      for seed in REPLICATION_SEEDS]
PARENT_KERNEL = 'carasik/rhpr-ml1m-mixed128-screen'
PARENT_COMMIT = '1074a7599c52875d66eb83716491f377c2310f33'
PARENT_TRIAL = 'trials/mixed128_merge120_42'
PARENT_FILES = {
    'run-manifest.json', 'index-report.json', 'prepared-cases/data-report.json',
    *(f'prepared-cases/{split}-cases.npz' for split in ('train','validation','test')),
    *(f'frozen-indexes/{name}.npy' for name in ('items-42','pair-keys','pairs-42')),
    f'{PARENT_TRIAL}/training-view-provenance.json', f'{PARENT_TRIAL}/trial-result.json',
}
MIXED_VIEWS = dict(full_probability=.5, unit='optimizer_batch', inference='fixed_pairs',
                   rng_stream=7301, preserve_native_positions=True)
POLICY = dict(history_items=128, merge_prefix_items=MERGE_PREFIXES,
    protected_tail_items=[64,32,16,8], training='fixed_oldest_valid_adjacent_pairs',
    padding='left_pad_compacted_3plus1_units', odd_prefix='keep_single_before_tail',
    pair_index='one_shared_train_pair_union', suffix='train_then_validation_then_test',
    pair_text_labels=['First movie','Second movie'], pair_max_tokens=2048,
    all_targets_single=True, random_training=False, fixed_epochs=40)
MIXED_POLICY = {**POLICY, 'training': 'full_or_fixed_pairs_per_optimizer_batch',
                'random_training': True}


def validate(c):
    from rhpr.gr.ml1m_last8 import WINDOWS, TRIALS as BASE_TRIALS, validate as base_validate
    t = c['training']; reference = c['reference_control']
    mixed = c.get('mixed_views')
    parent = c.get('frozen_parent')
    trials = REPLICATION_TRIALS if parent else MIXED_TRIALS if mixed else TRIALS
    if (('mixed_views' in c and mixed != MIXED_VIEWS)
            or c['kind'] != 'gr_ml1m_last8_pairs' or c['windows'] != [128] or c['trials'] != trials
            or c['policy'] != (MIXED_POLICY if mixed else POLICY)
            or c['expected_sid_file_sha256'] != REFERENCE_SID_SHA256
            or (t['min_epochs'],t['max_epochs'],t['max_seconds']) != (40,40,7200)
            or c['pair_batches'] != dict(physical=64,evaluation=16)
            or c['pair_embedding_batch_size'] != 128
            or reference['kernel'] != 'carasik/rhpr-ml1m-last8-windows/2'
            or reference['source_commit'] != '5c1af46bddb1a848877acead858ec855e77201ed'
            or reference['best_epoch'] != 35 or reference['matched_max_epoch'] != 40
            or not re.fullmatch('[0-9a-f]{64}',c['expected_pair_initial_state_sha256'])
            or c['suite_deadline_seconds'] != (7200 if parent else 14400)):
        raise ValueError('Pair128 configuration differs from the registered 40-epoch protocols')
    if parent:
        initializations = c['replication_initializers']
        if (mixed != MIXED_VIEWS or c['seeds'] != REPLICATION_SEEDS
                or parent['kernel'] != PARENT_KERNEL or parent['git_sha'] != PARENT_COMMIT
                or c['kaggle']['kernel_sources'] != [PARENT_KERNEL]
                or set(parent['files']) != PARENT_FILES
                or parent['files']['frozen-indexes/items-42.npy']['sha256'] != REFERENCE_SID_SHA256
                or set(initializations) != {str(seed) for seed in REPLICATION_SEEDS}
                or any(set(v) != {'base','pair'} or
                       any(re.fullmatch('[0-9a-f]{64}', h) is None for h in v.values())
                       for v in initializations.values())
                or len({v['pair'] for v in initializations.values()} | {c['expected_pair_initial_state_sha256']}) != 3):
            raise ValueError('expected only two new Mixed120 seeds on the completed frozen index')
    if (len(reference['validation_curve']) != 8
            or [r['epoch'] for r in reference['validation_curve']] != list(range(5,41,5))
            or max(reference['validation_curve'],key=lambda r:r['ndcg10'])['epoch'] != reference['best_epoch']
            or reference['validation_ndcg10'] != max(r['ndcg10'] for r in reference['validation_curve'])):
        raise ValueError('existing control checkpoint must also be the best within40epochs')
    normalized=copy.deepcopy(c);normalized.update(kind='gr_ml1m_last8',windows=WINDOWS,trials=BASE_TRIALS)
    normalized['training']['max_epochs']=120
    if parent:
        normalized['seeds'] = [42]
        normalized['kaggle']['kernel_sources'] = []
    base_validate(normalized)


def checked_contexts(contexts, nitems):
    h = np.asarray(contexts)
    if h.ndim != 2 or h.shape[1] < 128 or h.dtype.kind not in 'iu': raise ValueError('integer histories with128slots required')
    h=h[:, -128:]; n=(h >= 0).sum(1)
    if (np.any(h < -1) or np.any(h >= nitems)
            or np.any((np.arange(128)[None,:] < 128-n[:,None]) != (h < 0))):
        raise ValueError('invalid catalogue ID or noncontiguous left padding')
    return h,n


def pair_keys(contexts, nitems, merge_items):
    if merge_items not in MERGE_PREFIXES: raise ValueError('unregistered merge prefix')
    h,n=checked_contexts(contexts,nitems);tail=128-merge_items;parts=[]
    for j in range(merge_items//2):
        rows=np.flatnonzero(2*j+1 < np.maximum(n-tail,0))
        if len(rows):
            pos=128-n[rows]+2*j
            parts.append(h[rows,pos].astype(np.int64)*nitems+h[rows,pos+1])
    return np.unique(np.concatenate(parts)) if parts else np.empty(0,dtype=np.int64)


def encode(contexts, sids, keys, pairs, merge_items):
    """Preserve all valid events, the protected recent tail and every suffix."""
    if (merge_items not in MERGE_PREFIXES or sids.shape != (len(sids),4)
            or pairs.shape != (len(keys),4) or np.any(keys[1:] <= keys[:-1])):
        raise ValueError('expected sorted pair keys and two3+1SIDtables')
    h,n=checked_contexts(contexts,len(sids));p=np.maximum(n-(128-merge_items),0)//2
    capacity=128-merge_items//2;units=n-p
    out=np.full((len(h),capacity,4),-1,dtype=np.int32)
    for col in range(capacity):
        local=col-(capacity-units);valid=local >= 0;ispair=valid & (local < p)
        source=128-n+np.where(local < p,2*local,local+p)
        rows=np.flatnonzero(valid & ~ispair)
        if len(rows):out[rows,col]=sids[h[rows,source[rows]]]+np.arange(4)*256
        rows=np.flatnonzero(ispair)
        if len(rows):
            wanted=h[rows,source[rows]].astype(np.int64)*len(sids)+h[rows,source[rows]+1]
            at=np.searchsorted(keys,wanted)
            if np.any(at >= len(keys)) or not np.array_equal(keys[at],wanted): raise ValueError('missing pair SID; never drop events')
            out[rows,col]=pairs[at]+(4+np.arange(4))*256
    return out.reshape(len(h),capacity*4)


def cache_sets(data):
    from rhpr.gr.ml100k_pair_raw import cases
    # Smaller protected tails contain every other variant's oldest pairs.
    return {s:pair_keys(cases(data,s)[0],3706,120) for s in ('train','validation','test')}


def full_context_selector(contexts, sids, pair_contexts, config, merge_items):
    """Switch between the same histories at their respective native widths."""
    from rhpr.gr.kindle_mixed_views import MixedViews

    history, _ = checked_contexts(contexts, len(sids))
    tokens = sids[np.maximum(history, 0)] + 256 * np.arange(4, dtype=np.int32)
    full = np.where(history[:, :, None] >= 0, tokens, -1).reshape(len(history), 512)
    return MixedViews(full, pair_contexts, batch_size=config['training']['batch_size'],
                      seed=config['seed'], full_probability=config['mixed_views']['full_probability'],
                      layout=(('full128', 512), ('pair128', 4 * (128 - merge_items // 2))))


def profiling_batch(index, contexts, users, targets, ignored_sids, device):
    """Profiling's five-argument builder contract, using the preencoded index."""
    return index.model_batch(contexts,users,targets,device)


def frozen_inputs(c, input_root=Path('/kaggle/input')):
    """Load pinned cases and tokens from the completed Mixed128 pilot."""
    from rhpr.gr.ml100k_pair_raw import cases
    from rhpr.gr.pair_test import locate_parent

    parent = locate_parent(c['frozen_parent'], input_root)
    original = json.loads((parent/'run-manifest.json').read_text())['config']
    shared_keys = ('data','training','generator','quantizer','policy','mixed_views','sampling',
                   'evaluation','pair_batches','reference_control','expected_raw_case_sha256')
    if original['trials'] != MIXED_TRIALS or any(original[k] != c[k] for k in shared_keys):
        raise ValueError('frozen parent uses a different data, model or training protocol')
    data, indexes = parent/'prepared-cases', parent/'frozen-indexes'
    report = json.loads((data/'data-report.json').read_text())
    for split, expected in c['expected_raw_case_sha256'].items():
        if case_digest(cases(data, split)) != expected:
            raise ValueError(f'parent case data changed: {split}')
    prior = parent/PARENT_TRIAL
    reference = json.loads((prior/'trial-result.json').read_text())
    views = json.loads((prior/'training-view-provenance.json').read_text())
    if (reference['seed'] != 42 or reference['merge_items'] != 120
            or reference['selection']['training_epochs'] != 40
            or reference['pair_sid_file_sha256'] != c['frozen_parent']['files']['frozen-indexes/pairs-42.npy']['sha256']):
        raise ValueError('frozen parent lacks the completed seed 42 Mixed120 comparison')
    return parent, data, indexes, report, reference, views


def run_trial(trial,p):
    import torch
    from rhpr.gr.grid import build_grid_generator
    from rhpr.gr.ml100k_pair_raw import cases
    from rhpr.gr.pair_unique4 import attach4,InputIndex4
    from rhpr.gr.profiling import benchmark_training,ProfileDeadline
    from rhpr.gr.sid import SidIndex
    from rhpr.gr.sid_study import state_digest
    c=copy.deepcopy(p['config']);c['active_trial']=trial;c['data']['history_items']=128
    seed = trial['seed']
    c['seed'] = seed
    expected_init = c.get('replication_initializers', {}).get(str(seed), {
        'base': c['expected_initial_state_sha256'], 'pair': c['expected_pair_initial_state_sha256']})
    c['training'].update(physical_batch_size=c['pair_batches']['physical'],evaluation_batch_size=c['pair_batches']['evaluation'])
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    root=Path(p['indexes']);sids=np.load(root/'items-42.npy');keys=np.load(root/'pair-keys.npy');pairs=np.load(root/'pairs-42.npy')
    index=SidIndex.from_full_sids(sids,width=256);wrapped=InputIndex4(index)
    model=build_grid_generator(sids,c['generator'],seed,'cpu',p['source'])
    base=state_digest(dict(model.named_parameters()))
    if base != expected_init['base']: raise ValueError('base parameters differ from CPU preflight')
    before={k:v.detach().clone() for k,v in model.named_parameters()};attach4(model)
    if any(not torch.equal(v,dict(model.named_parameters())[k]) for k,v in before.items()): raise ValueError('pair adapter changed common parameters')
    del before
    initial=state_digest(dict(model.named_parameters()))
    if initial != expected_init['pair']: raise ValueError('pair initialization differs from CPU preflight')
    m=trial['merge_items'];raw={s:cases(p['data'],s) for s in ('train','validation')}
    encoded={s:(encode(h,sids,keys,pairs,m),u,t) for s,(h,u,t) in raw.items()}
    views = None
    if 'mixed_views' in c:
        views = full_context_selector(raw['train'][0], sids, encoded['train'][0], c, m)
    out=Path(p['output'])/'trials'/trial['id'];out.mkdir(parents=True)
    if views is not None:
        provenance = dict(
            policy=c['mixed_views'], native_sid_widths=views.widths,
            full_case_sha256=case_digest((views.full, *encoded['train'][1:])),
            pair_case_sha256=case_digest(encoded['train']))
        if p.get('reference_views', provenance) != provenance:
            raise ValueError('encoded training views differ from seed 42')
        _write(out/'training-view-provenance.json', provenance)
    model.eval()
    with torch.inference_mode(),torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH),torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU],with_flops=True) as prof:
        loss=_loss(model,wrapped.model_batch(*(v[:1] for v in encoded['validation']),'cpu'))
    if not torch.isfinite(loss):raise FloatingPointError('nonfinite CPU preflight')
    forward=dict(parameters=sum(v.numel() for v in model.parameters()),added_pair_parameters=4*256*128,
        encoder_positions=encoded['validation'][0].shape[1]+1,supported_forward_flops_batch1=sum(e.flops for e in prof.key_averages()),
        scope='B1 fixed-padded CPU mathSDPA teacher-forced4code forward,FMA2; excludes beam/backward/text encoding')
    _write(out/'forward-flops.json',forward);model.to('cuda:0')
    compiled=torch.compile(model,**c['training']['compile_training'])
    profile=benchmark_training(compiled,encoded['train'],sids,np.random.default_rng(42).permutation(48320)[:512],
        c['training']['physical_batch_size'],'fp16','full_pipeline',c['training'],dict(seed=42,warmup=1,iterations=2),
        ProfileDeadline(180),'cuda:0',batch_builder=partial(profiling_batch,wrapped))
    if state_digest(dict(model.named_parameters())) != initial:raise RuntimeError('profile changed initial weights')
    _write(out/'training-profile.json',profile);_event('ml1m_pairs_profile_complete',trial=trial['id'],**profile)
    if views is not None:
        full_profile = benchmark_training(
            compiled, (views.full, *encoded['train'][1:]), sids,
            np.random.default_rng(42).permutation(48320)[:512], c['training']['physical_batch_size'],
            'fp16', 'full_pipeline', c['training'], dict(seed=42, warmup=1, iterations=2),
            ProfileDeadline(180), 'cuda:0', batch_builder=partial(profiling_batch, wrapped))
        if state_digest(dict(model.named_parameters())) != initial:
            raise RuntimeError('full-view profile changed initial weights')
        _write(out/'training-profile-full.json', full_profile)
        _event('ml1m_pairs_full_profile_complete', trial=trial['id'], **full_profile)
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.cuda.reset_peak_memory_stats();started=time.monotonic()
    def builder(root,width,split):
        if split not in encoded:raise ValueError('TEST forbidden during training')
        return encoded[split]
    selection=_train_generator(model,p['data'],wrapped,c,out,'cuda:0',example_builder=builder,
                               training_context_selector=views)
    if (selection['training_epochs'] != 40 or selection['validation_checks'] != 8
            or selection['optimizer_updates']+selection['amp_skipped_updates'] != 3800):
        raise RuntimeError('incomplete40epochscreen; preserve checkpoints')
    curve=json.loads((out/'training-curves.json').read_text())
    timing=dict(wall_seconds=time.monotonic()-started,warm_epoch_training_median_seconds=float(np.median([v['train_seconds'] for v in curve[1:]])),
        first_epoch_training_seconds=curve[0]['train_seconds'],cuda_peak_allocated_bytes=int(torch.cuda.max_memory_allocated()),cuda_peak_reserved_bytes=int(torch.cuda.max_memory_reserved()))
    h,u,t=cases(p['data'],'test');x=encode(h,sids,keys,pairs,m);checkpoint=out/'generator-best.pt'
    ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if ck['epoch'] != selection['best_epoch']:raise ValueError('selected checkpoint mismatch')
    model.load_state_dict(ck['model'],strict=True)
    metrics,pred,scores=_evaluate(model,(x,u,t),wrapped,c['training']['evaluation_batch_size'],'cuda:0','fp32')
    np.savez_compressed(out/'test-best-predictions.npz',semantic_ids=pred,scores=scores,user_ids=u,target_item_ids=t)
    test=dict(epoch=selection['best_epoch'],metrics=metrics,checkpoint_sha256=file_sha256(checkpoint),selection='raw validation NDCG@10 only',seen_items_filtered=False)
    _write(out/'test-metrics.json',test)
    row=dict(trial=trial['id'],seed=seed,merge_items=m,protected_tail=128-m,base_initial_state_sha256=base,initial_state_sha256=initial,
        sid_file_sha256=file_sha256(root/'items-42.npy'),pair_sid_file_sha256=file_sha256(root/'pairs-42.npy'),
        raw_case_sha256={s:case_digest(v) for s,v in raw.items()},case_targets_sha256={s:case_digest(v[1:]) for s,v in raw.items()},
        test_case_sha256=case_digest((h,u,t)),selection=selection,forward=forward,timing=timing,profile=profile,test=test,
        epoch_capped_screen=True,training_pair_view='fixed',test_pair_view='fixed_oldest_valid_pairs')
    if views is not None:
        row.update(training_pair_view='batch_mixture', mixed_views=c['mixed_views'],
                   full_training_profile=full_profile,
                   train_view_examples={name: sum(v['training_views']['examples'][name] for v in curve)
                                        for name in views.widths})
    _write(out/'trial-result.json',row);return row


def worker(plan_path,slot):
    import torch,transformers
    p=json.loads(Path(plan_path).read_text());validate(p['config']);torch.set_num_threads(1)
    if torch.cuda.device_count()!=1 or torch.cuda.get_device_capability()!=(7,5) or transformers.__version__!='5.22.1':raise RuntimeError('requires isolated T4/HF5.22.1')
    status=dict(results={},failures=[])
    for trial in p['assignments'][slot]['trials']:
        _event('ml1m_pairs_trial_start',trial=trial['id'])
        try:
            row=run_trial(trial,p);status['results'][trial['id']]=row;_event('ml1m_pairs_trial_complete',**row)
        except Exception as error:
            out=Path(p['output'])/'trials'/trial['id'];out.mkdir(parents=True,exist_ok=True)
            (out/'failure.log').write_text(traceback.format_exc());status['failures'].append(dict(trial=trial['id'],exception_type=type(error).__name__,message=str(error)))
            _event('ml1m_pairs_trial_failed',**status['failures'][-1])
        _write(Path(p['output'])/'workers'/f'worker-{slot}.json',status)
        torch._dynamo.reset();gc.collect();torch.cuda.empty_cache()


def prepare_indexes(c, output, source):
    import torch
    from sentence_transformers import SentenceTransformer
    from rhpr.gr.grid import fit_grid_rkmeans
    from rhpr.gr.kindle_pairs import encode_pair_cache
    from rhpr.gr.pair_sid import predict_rq
    from rhpr.gr.pair_unique4 import unique_pair_sids
    from rhpr.gr.sid import SidIndex
    from rhpr.offline_runtime import resolve_dataset_directory
    resources = resolve_dataset_directory(c['resources']['path'])
    archive = resolve_dataset_directory(c['movie_input']['path'])/'ml-1m-source.bin'
    if file_sha256(archive)!=c['movie_input']['source_zip_sha256']:raise ValueError('raw source changed')
    data=output/'prepared-cases';texts,report=prepare_raw(archive,data,c['data'])
    if {s:v['raw_case_sha256'] for s,v in report['splits'].items()} != c['expected_raw_case_sha256']:raise ValueError('raw cases differ')
    sets=cache_sets(data);keys=np.unique(np.concatenate(list(sets.values())));mask=np.isin(keys,sets['train'])
    if {s:len(k) for s,k in sets.items()} != c['expected_pair_counts'] or len(keys)!=c['expected_pair_union']:raise ValueError('pair schedule differs from preflight')
    cache=output/'pair-cache';cache.mkdir();indexes=output/'frozen-indexes';indexes.mkdir()
    np.save(indexes/'pair-keys.npy',keys);np.save(cache/'train-mask.npy',mask)
    _event('ml1m_pairs_embedding_start',items=len(texts),pair_counts={s:len(v) for s,v in sets.items()},pair_union=len(keys))
    encoder=SentenceTransformer(str(resources/'sentence-t5-base'),device='cuda:0',local_files_only=True);encoder.max_seq_length=256;encoder.eval()
    vec=np.asarray(encoder.encode(texts,batch_size=32,convert_to_numpy=True,show_progress_bar=False),dtype=np.float32)
    np.save(cache/'item-embeddings.npy',vec)
    codes,centers,item_fit=fit_grid_rkmeans(vec,c['quantizer'],42,'cuda:0',source)
    item_index=SidIndex(codes,width=256);np.save(indexes/'items-42.npy',item_index.sids);np.save(indexes/'item-centroids-42.npy',centers)
    if file_sha256(indexes/'items-42.npy') != c['expected_sid_file_sha256']:raise ValueError('single-item index differs from control; no pair embedding or training')
    encoder.max_seq_length=POLICY['pair_max_tokens'];encode_pair_cache(encoder,texts,keys,cache/'pair-embeddings.npy',POLICY['pair_text_labels'],
        batch_size=c['pair_embedding_batch_size'],event_prefix='ml1m_pairs')
    del encoder;gc.collect();torch.cuda.empty_cache()
    pairvec=np.load(cache/'pair-embeddings.npy',mmap_mode='r')
    fitcodes,centers,pair_fit=fit_grid_rkmeans(np.asarray(pairvec[mask]),c['quantizer'],42,'cuda:0',source)
    codes=predict_rq(pairvec,centers,'cuda:0')
    if not np.array_equal(codes[mask],fitcodes):raise ValueError('pair quantizer assignment mismatch')
    # Preserve the fitted index even if the collision-suffix guard rejects it.
    np.save(indexes/'pair-codes-42.npy',codes);np.save(indexes/'pair-centroids-42.npy',centers)
    collision_counts=np.unique(codes,axis=0,return_counts=True)[1]
    _write(indexes/'pair-fit-report.json',dict(fit=pair_fit,semantic_groups=len(collision_counts),max_collision_group=int(collision_counts.max()),pair_union=len(keys)))
    pairs=unique_pair_sids(keys,codes,sets['train'],np.union1d(sets['train'],sets['validation']))
    np.save(indexes/'pairs-42.npy',pairs)
    _write(output/'index-report.json',dict(item=item_fit,pair=pair_fit,pair_counts={s:len(v) for s,v in sets.items()},pair_union=len(keys),
        max_pair_suffix=int(pairs[:,3].max()),unique_pair_sids=len(np.unique(pairs,axis=0)),pair_fit_on_train_only=True,
        suffix_policy=POLICY['suffix'],all_targets_single=True,pair_quantizer_shared_across_variants=True))
    del pairvec,vec,codes,centers,pairs,fitcodes;gc.collect();torch.cuda.empty_cache()
    return data, indexes, report


def run(c,output):
    import torch
    from rhpr.gr.ae_suite import assignments,launch_workers
    from rhpr.gr.grid import prepare_grid_source
    from rhpr.gr.resources import unpack_grid,verify_resources
    from rhpr.offline_runtime import resolve_dataset_directory

    validate(c);output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True);(output/'workers').mkdir()
    torch.set_num_threads(1)
    if torch.cuda.device_count()!=2 or any(torch.cuda.get_device_capability(i)!=(7,5) for i in (0,1)):raise RuntimeError('requires two T4s; no fallback')
    roots={name:resolve_dataset_directory(c[name]['path']) for name in ('resources','modern_runtime')}
    for name in roots:verify_resources(roots[name],c[name]['manifest_sha256'])
    scratch=Path(c['resources']['scratch_path']);scratch.mkdir(parents=True,exist_ok=True)
    source=prepare_grid_source(unpack_grid(roots['resources']/'grid-source.bin',scratch/'original'),scratch/'grid')
    extra = {}
    if 'frozen_parent' in c:
        parent, data, indexes, report, reference, views = frozen_inputs(c)
        extra = dict(reference_views=views, reference_mixed42=reference)
        _write(output/'index-report.json', json.loads((parent/'index-report.json').read_text()))
        _write(output/'frozen-parent-verification.json', c['frozen_parent'])
        _event('ml1m_pairs_frozen_parent_verified', kernel=c['frozen_parent']['kernel'],
               revision=c['frozen_parent']['git_sha'], files=len(c['frozen_parent']['files']))
    else:
        data, indexes, report = prepare_indexes(c, output, source)
    _write(output/'reference-control.json',c['reference_control'])
    overlay=scratch/'modern-site';overlay.mkdir()
    with (output/'modern-install.log').open('w') as log:
        subprocess.run([sys.executable,'-m','pip','install','--no-index','--no-deps','--no-cache-dir','--no-compile','--require-hashes',
            '--only-binary=:all:','--target',str(overlay),'--find-links',str(roots['modern_runtime']),'-r',str(roots['modern_runtime']/'requirements-lock.txt')],
            stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)
    sys.path.insert(0,str(overlay));p=dict(config=c,output=str(output),source=str(source),data=str(data),indexes=str(indexes),assignments=assignments(c,2),**extra)
    path=output/'plan.json';_write(path,p);summary=launch_workers(path,worker_module='rhpr.gr.ml1m_last8_pairs',event_prefix='ml1m_pairs')
    rows=list(summary['results'].values())
    shared = ('sid_file_sha256','pair_sid_file_sha256','raw_case_sha256','case_targets_sha256','test_case_sha256')
    if extra:
        keys = (*shared, 'forward')
        anchor = extra['reference_mixed42']
    else:
        keys = (*shared, 'base_initial_state_sha256', 'initial_state_sha256')
        anchor = rows[0]
    for key in keys:
        if any(r[key]!=anchor[key] for r in rows):raise ValueError('unmatched pair comparison: '+key)
    _write(output/'study-summary.json',dict(results=summary['results'],data=report,reference_control=c['reference_control'],
        new_generator_fits=len(rows),epoch_capped_screen=True,**extra))
    metrics=dict(experiment=c['name'],models=len(rows),epochs=40,test_evaluated=True)
    if not extra:
        metrics['seed'] = 42
    for name,row in summary['results'].items():
        metrics[name+'_validation_ndcg10']=row['selection']['validation_ndcg10'];metrics[name+'_test_ndcg10']=row['test']['metrics']['ndcg@10']
        metrics[name+'_forward_flops']=row['forward']['supported_forward_flops_batch1']
    return metrics


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--slot',type=int,required=True)
    a=p.parse_args();worker(a.plan,a.slot)
