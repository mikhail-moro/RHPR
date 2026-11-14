"""Two long-window pair fits with frozen item SIDs and read-only control TESTs."""
from __future__ import annotations

import argparse
import copy
import gc
import json
import random
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from rhpr.gr.experiment import _evaluate, _event, _loss, _train_generator, _write

TRIALS = [dict(id=f'pair{w}_wide_42', seed=42, history_items=w) for w in (64, 32)]


def validate(c):
    from rhpr.gr import kindle_capacity, kindle_windows
    if (c['kind'] != 'gr_kindle_pairs' or c['trials'] != TRIALS
        or c['seeds'] != [42] or c['windows'] != [32, 64]
        or not c['evaluation']['test_after_selection']
        or c['pair_text_labels'] != ['First book', 'Second book']
        or c['suffix_policy'] != 'train_then_val_then_test_sorted_pair_keys'
        or c['parent']['kernel'] != 'carasik/rhpr-kindle-capacity-pilot'
        or c['parent']['git_sha'] != '6d2919a155893e4c6bbb859a1fbcc0c800573c6c'
        or c['kaggle']['kernel_sources'] != [c['parent']['kernel']]
        or c['control_tests'] != [20, 32]):
        raise ValueError('expected only Pair32/64 seed42 and two read-only control TESTs')
    normalized = copy.deepcopy(c)
    normalized.update(kind='gr_kindle_windows', trials=kindle_windows.TRIALS,
                      seeds=kindle_windows.SEEDS, windows=kindle_windows.WINDOWS)
    normalized['generator'].update(kindle_capacity.NARROW)
    normalized['evaluation']['test_after_selection'] = False
    kindle_windows.validate(normalized)
    if any(c['generator'][k] != v for k, v in kindle_capacity.WIDE.items()):
        raise ValueError('use the existing wide generator without retuning')


def check_parent(c, old):
    for name in ('training', 'generator', 'data', 'movie_input', 'quantizer'):
        if c[name] != old[name]:
            raise ValueError('pair comparison changes ' + name)
    if c['evaluation'] != {**old['evaluation'], 'test_after_selection': True}:
        raise ValueError('only final TEST evaluation may change')


def collect_keys(contexts, width, nitems):
    """Ordered nonoverlapping pairs of the older half of each valid window."""
    h = np.asarray(contexts)[:, -width:]
    if width not in (32, 64) or np.any(h < -1) or np.any(h >= nitems):
        raise ValueError('invalid item history')
    lengths = (h >= 0).sum(1)
    rows = np.arange(len(h))
    parts = []
    for j in range(width // 4):
        mask = 2*j + 1 < lengths // 2
        positions = width - lengths[mask] + 2*j
        parts.append(h[rows[mask], positions].astype(np.int64)*nitems
                     + h[rows[mask], positions+1])
    return np.unique(np.concatenate(parts))


def encode(contexts, item_sids, keys, pair_sids, width):
    """Each pair AND each single uses 3+1; padding never enters a pair."""
    if width not in (32, 64) or item_sids.shape[1] != 4 or pair_sids.shape != (len(keys), 4):
        raise ValueError('expected 32/64 items and unique four-code SIDs')
    nitems = len(item_sids)
    ncols = 4 * max(n - n//4 for n in range(width+1))
    out = np.full((len(contexts), ncols), -1, dtype=np.int32)
    for rowno, row in enumerate(np.asarray(contexts)[:, -width:]):
        h = row[row >= 0]
        if np.any(h >= nitems):
            raise ValueError('item out of catalogue')
        old = len(h)//2
        tokens = []
        j = 0
        while j+1 < old:
            key = int(h[j])*nitems + int(h[j+1])
            pos = int(np.searchsorted(keys, key))
            if pos == len(keys) or keys[pos] != key:
                raise ValueError('pair absent from frozen index')
            tokens.extend(pair_sids[pos] + (4+np.arange(4))*256)
            j += 2
        while j < len(h):
            tokens.extend(item_sids[h[j]] + np.arange(4)*256)
            j += 1
        if tokens:
            out[rowno, -len(tokens):] = tokens
    return out


def encode_pair_cache(encoder, texts, keys, path, labels):
    nitems = len(texts)
    dest = np.lib.format.open_memmap(path, mode='w+', dtype='float32', shape=(len(keys), 768))
    started = time.monotonic()
    for begin in range(0, len(keys), 2048):
        chunk = keys[begin:begin+2048]
        rows = [f'{labels[0]}: {texts[int(k)//nitems]} {labels[1]}: {texts[int(k)%nitems]}' for k in chunk]
        values = np.asarray(encoder.encode(rows, batch_size=32, convert_to_numpy=True,
                                          show_progress_bar=False), dtype=np.float32)
        if values.shape != (len(chunk), 768) or not np.isfinite(values).all():
            raise ValueError('invalid pair embeddings')
        dest[begin:begin+len(chunk)] = values
        dest.flush()
        _event('kindle_pairs_embedding_progress', completed=begin+len(chunk), total=len(keys),
               seconds=time.monotonic()-started)
    del dest


def test_selected(model, examples, wrapped, path, checkpoint, expected_epoch):
    """No fit, search or checkpoint choice depends on TEST."""
    import torch
    from rhpr.gr.data import file_sha256
    ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if ck['epoch'] != expected_epoch:
        raise ValueError('selected checkpoint epoch differs')
    model.load_state_dict(ck['model'], strict=True)
    metrics, ids, scores = _evaluate(model, examples, wrapped, 64, 'cuda:0', 'fp32')
    np.savez_compressed(path/'test-best-predictions.npz', semantic_ids=ids, scores=scores,
                        user_ids=examples[1], target_item_ids=examples[2])
    result = dict(epoch=expected_epoch, metrics=metrics, checkpoint_sha256=file_sha256(checkpoint),
                  selection='raw validation NDCG@10 only', seen_items_filtered=False)
    _write(path/'test-metrics.json', result)
    return result


def run_trial(trial, p):
    import torch
    from rhpr.gr.data import file_sha256
    from rhpr.gr.grid import build_grid_generator
    from rhpr.gr.kindle_windows import encode_window
    from rhpr.gr.ml100k_pair_raw import cases
    from rhpr.gr.movie_data import case_digest
    from rhpr.gr.pair_unique4 import attach4, InputIndex4
    from rhpr.gr.sid import SidIndex
    from rhpr.gr.sid_study import state_digest
    c = copy.deepcopy(p['config'])
    width = trial['history_items']
    c['data']['history_items'] = width
    random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    root = Path(p['indexes'])
    idx = SidIndex.from_full_sids(np.load(root/'items-42.npy'), width=256)
    keys = np.load(root/'pair-keys.npy'); pairs = np.load(root/'pairs-42.npy')
    wrapped = InputIndex4(idx)
    model = attach4(build_grid_generator(idx.sids, c['generator'], 42, 'cpu', p['source']))
    initial = state_digest(dict(model.named_parameters()))
    baseline = json.loads((Path(p['parent'])/'trials/recent32_wide_42/trial-result.json').read_text())
    if initial != baseline['initial_state_sha256']:
        raise ValueError('pair initializer differs from existing controls')
    out = Path(p['output'])/'trials'/trial['id']; out.mkdir(parents=True)
    raw = {s: cases(p['data'], s) for s in ('train', 'validation')}
    encoded = {s: (encode(h, idx.sids, keys, pairs, width), u, t) for s, (h,u,t) in raw.items()}
    probe = tuple(a[:1] for a in encoded['validation'])
    model.eval()
    with torch.inference_mode(), torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH), torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], with_flops=True) as prof:
        _loss(model, wrapped.model_batch(*probe, 'cpu'))
    forward = dict(parameters=sum(v.numel() for v in model.parameters()),
        encoder_positions=encoded['validation'][0].shape[1]+1,
        supported_forward_flops_batch1=sum(e.flops for e in prof.key_averages()),
        scope=baseline['forward']['scope'])
    if initial != state_digest(dict(model.named_parameters())):
        raise RuntimeError('profiling changed weights')
    _write(out/'forward-flops.json', forward)
    model.to('cuda:0'); torch.cuda.reset_peak_memory_stats(); started=time.monotonic()
    def builder(root, ignored, split):
        if split not in encoded:
            raise ValueError('TEST forbidden during training')
        return encoded[split]
    selection = _train_generator(model, p['data'], wrapped, c, out, 'cuda:0', example_builder=builder)
    epochs = selection['training_epochs']
    if (not 40 <= epochs <= 60 or selection['validation_checks'] != epochs//5
        or selection['stopping_reason'] not in ('epoch_limit', 'early_stopping')
        or selection['optimizer_updates']+selection['amp_skipped_updates'] != ((len(raw['train'][2])+511)//512)*epochs):
        raise RuntimeError('incomplete declared fit')
    curves=json.loads((out/'training-curves.json').read_text())
    timing=dict(wall_seconds=time.monotonic()-started,
        first_epoch_training_seconds=curves[0]['train_seconds'],
        warm_epoch_training_median_seconds=float(np.median([r['train_seconds'] for r in curves[1:]])),
        cuda_peak_allocated_bytes=int(torch.cuda.max_memory_allocated()),
        cuda_peak_reserved_bytes=int(torch.cuda.max_memory_reserved()))
    h,u,t=cases(p['data'],'test')
    test=test_selected(model,(encode(h,idx.sids,keys,pairs,width),u,t),wrapped,out,
                       out/'generator-best.pt',selection['best_epoch'])
    row=dict(trial=trial['id'], seed=42, history_items=width, mode='pair',
        initial_state_sha256=initial, sid_file_sha256=file_sha256(root/'items-42.npy'),
        pair_sid_file_sha256=file_sha256(root/'pairs-42.npy'), selection=selection, forward=forward,
        raw_case_sha256={s:case_digest(a) for s,a in raw.items()},
        case_targets_sha256={s:case_digest(a[1:]) for s,a in raw.items()},
        test_case_sha256=case_digest((h,u,t)),timing=timing,test=test,test_evaluated=True)
    for key in ('sid_file_sha256','raw_case_sha256','case_targets_sha256'):
        if row[key] != baseline[key]:
            raise RuntimeError('baseline differs: '+key)
    _write(out/'trial-result.json',row)
    # Reuse one existing wide control per GPU, only its already selected weights.
    control_width=20 if width==64 else 32
    name=f'recent{control_width}_wide_42';parent=Path(p['parent'])/'trials'/name
    old=json.loads((parent/'trial-result.json').read_text())
    control_out=Path(p['output'])/'control-tests'/name;control_out.mkdir(parents=True)
    control_test=test_selected(model,(encode_window(h,idx.sids,control_width),u,t),wrapped,
                               control_out,parent/'generator-best.pt',old['selection']['best_epoch'])
    _write(control_out/'control-result.json',dict(parent_result=old,test=control_test,
           test_case_sha256=case_digest((h,u,t)),training_performed=False))
    return row


def worker(plan_path, slot):
    import torch, transformers
    p=json.loads(Path(plan_path).read_text());validate(p['config']);torch.set_num_threads(1)
    if torch.cuda.device_count()!=1 or torch.cuda.get_device_capability()!=(7,5) or transformers.__version__!='5.22.1':
        raise RuntimeError('requires one isolated T4 with HF5.22.1')
    state=dict(results={},failures=[])
    for trial in p['assignments'][slot]['trials']:
        _event('kindle_pairs_trial_start',trial=trial['id'],history_items=trial['history_items'])
        try:
            row=run_trial(trial,p);state['results'][trial['id']]=row
            _event('kindle_pairs_trial_complete',**row)
        except Exception as error:
            out=Path(p['output'])/'trials'/trial['id'];out.mkdir(parents=True,exist_ok=True)
            (out/'failure.log').write_text(traceback.format_exc())
            state['failures'].append(dict(trial=trial['id'],exception_type=type(error).__name__,message=str(error)))
            _event('kindle_pairs_trial_failed',**state['failures'][-1])
        _write(Path(p['output'])/'workers'/f'worker-{slot}.json',state)
        torch._dynamo.reset();gc.collect();torch.cuda.empty_cache()


def run(c, output):
    import torch
    from sentence_transformers import SentenceTransformer
    from rhpr.gr.ae_suite import assignments, launch_workers
    from rhpr.gr.data import file_sha256
    from rhpr.gr.grid import fit_grid_rkmeans, prepare_grid_source
    from rhpr.gr.ml100k_pair_raw import cases
    from rhpr.gr.pair_sid import predict_rq
    from rhpr.gr.pair_test import locate_parent
    from rhpr.gr.pair_unique4 import unique_pair_sids
    from rhpr.gr.resources import unpack_grid, verify_resources
    from rhpr.offline_runtime import resolve_dataset_directory
    validate(c);output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    (output/'workers').mkdir(exist_ok=True)
    if torch.cuda.device_count()!=2 or any(torch.cuda.get_device_capability(i)!=(7,5) for i in (0,1)):
        raise RuntimeError('requires two T4 GPUs; no fallback')
    torch.set_num_threads(1);parent=locate_parent(c['parent'])
    check_parent(c,json.loads((parent/'run-manifest.json').read_text())['config'])
    data=resolve_dataset_directory(c['movie_input']['path'])
    for name,digest in c['movie_input']['sha256'].items():
        if file_sha256(data/name)!=digest:raise ValueError('input digest mismatch: '+name)
    roots={}
    for name in ('resources','modern_runtime'):
        roots[name]=resolve_dataset_directory(c[name]['path']);verify_resources(roots[name],c[name]['manifest_sha256'])
    scratch=Path(c['resources']['scratch_path']);scratch.mkdir(parents=True,exist_ok=True)
    source=prepare_grid_source(unpack_grid(roots['resources']/'grid-source.bin',scratch/'original'),scratch/'grid')
    indexes=output/'frozen-indexes';indexes.mkdir();cache=output/'pair-cache';cache.mkdir()
    shutil.copy2(parent/'frozen-indexes/items-42.npy',indexes/'items-42.npy')
    baselines=output/'baselines';baselines.mkdir()
    for w in (20,32):
        name=f'recent{w}_wide_42';dest=baselines/name;dest.mkdir()
        for rel in c['parent']['files']:
            if rel.startswith(f'trials/{name}/') and not rel.endswith('.pt'):
                shutil.copy2(parent/rel,dest/Path(rel).name)
    sets={}
    for split in ('train','validation','test'):
        h,_,_=cases(data,split)
        sets[split]=np.union1d(*[collect_keys(h,w,c['data']['items']) for w in (32,64)])
    keys=np.unique(np.concatenate(list(sets.values())))
    trainkeys=sets['train'];oldkeys=np.union1d(trainkeys,sets['validation']);trainmask=np.isin(keys,trainkeys)
    np.save(indexes/'pair-keys.npy',keys);np.save(cache/'train-mask.npy',trainmask)
    _event('kindle_pairs_cache_start',counts={k:len(v) for k,v in sets.items()},union=len(keys))
    texts=json.loads((data/'texts.json').read_text())
    encoder=SentenceTransformer(str(roots['resources']/'sentence-t5-base'),device='cuda:0',local_files_only=True)
    encoder.max_seq_length=2048;encoder.eval()
    encode_pair_cache(encoder,texts,keys,cache/'pair-embeddings.npy',c['pair_text_labels'])
    del encoder;gc.collect();torch.cuda.empty_cache()
    vec=np.load(cache/'pair-embeddings.npy',mmap_mode='r')
    fitcodes,centers,report=fit_grid_rkmeans(np.asarray(vec[trainmask]),c['quantizer'],42,'cuda:0',source)
    codes=predict_rq(vec,centers,'cuda:0')
    if not np.array_equal(codes[trainmask],fitcodes):raise RuntimeError('pair code export mismatch')
    sids=unique_pair_sids(keys,codes,trainkeys,oldkeys)
    np.save(indexes/'pairs-42.npy',sids);np.save(indexes/'pair-centroids-42.npy',centers)
    _write(indexes/'index-42.json',dict(pair_fit=report,counts={k:len(v) for k,v in sets.items()},
        union=len(keys),unique_pair_sids=len(np.unique(sids,axis=0)),max_suffix=int(sids[:,3].max()),
        fit_on_train_pairs_only=True,item_index_refitted=False,windows_share_pair_index=True,
        test_pairs_only_in_observed_test_context=True,suffix_policy=c['suffix_policy'],
        template='First book: {title+categories} Second book: {title+categories}'))
    del vec,fitcodes,centers,codes,sids;gc.collect();torch.cuda.synchronize();torch.cuda.empty_cache()
    overlay=scratch/'modern-site';overlay.mkdir(parents=True,exist_ok=True)
    command=[sys.executable,'-m','pip','install','--no-index','--no-deps','--no-cache-dir','--no-compile',
        '--require-hashes','--only-binary=:all:','--target',str(overlay),'--find-links',str(roots['modern_runtime']),
        '-r',str(roots['modern_runtime']/'requirements-lock.txt')]
    with (output/'modern-install.log').open('w') as log:
        subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)
    plan=dict(config=c,output=str(output),source=str(source),data=str(data),indexes=str(indexes),
              parent=str(parent),assignments=assignments(c,2))
    _write(output/'plan.json',plan);sys.path.insert(0,str(overlay))
    summary=launch_workers(output/'plan.json',worker_module='rhpr.gr.kindle_pairs',event_prefix='kindle_pairs')
    tests={}
    for w in (20,32):
        name=f'recent{w}_wide_42'
        tests[name]=json.loads((output/'control-tests'/name/'control-result.json').read_text())
    _write(output/'study-summary.json',dict(results=summary['results'],control_tests=tests,parent=c['parent'],
        test_evaluated=True,new_generator_fits=2,item_tokenizer_refitted=False,control64_trained=False))
    return dict(experiment=c['name'],models=2,read_only_control_tests=2,test_evaluated=True,control64_trained=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--plan',required=True);parser.add_argument('--slot',type=int,required=True)
    args=parser.parse_args();worker(args.plan,args.slot)
