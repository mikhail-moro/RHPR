"""Read-only pair-boundary and full-single replay of frozen Kindle models."""
from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
import traceback
from pathlib import Path

import numpy as np

from rhpr.gr.experiment import _evaluate, _event, _write

VARIANTS = ('32_reference', '32_edges', '30_first', '30_last',
            '34_first', '34_last', '32_single')
TRIALS = [dict(id=m, seed=42) for m in ('mixed32', 'pair32')]
POSITIONS = dict(zip(VARIANTS, (97, 101, 93, 93, 105, 105, 129)))


def validate(c):
    if (c['kind'] != 'gr_kindle_alignment' or c['trials'] != TRIALS
        or c['seed'] != 42 or c['variants'] != list(VARIANTS)
        or c['expected_new_pairs'] != 838
        or c['training'] != dict(device='cuda', cpu_threads=1,
            evaluation_batch_size=64, evaluation_precision='fp32')
        or c['evaluation'] != dict(splits=['validation', 'test'], selected_epoch=55,
            training_forbidden=True, seen_items_filtered=False,
            selection_policy='all_predeclared_variants_no_test_selection')
        or c['execution'] != dict(worker_devices=['cuda:0', 'cuda:1'], cpu_threads_per_worker=1)
        or set(c['parents']) != {'pair', 'mixed'}
        or c['parents']['pair']['git_sha'] != '33754d70acc0e4cf826f139202632bf0ffe6ac5b'
        or c['parents']['mixed']['git_sha'] != '88722e9081aae120e9af261f550d50fe74864da6'
        or c['kaggle']['kernel_sources'] != [c['parents'][k]['kernel'] for k in ('pair', 'mixed')]
        or c['generator']['beam_width'] != 20 or c['generator']['attn_implementation'] != 'sdpa'):
        raise ValueError('expected frozen Mixed/Pair55 seven layouts on VAL and TEST only')


def segments(row, variant):
    """Preserve chronological order; no dropped, duplicated or future events."""
    if variant not in VARIANTS:
        raise ValueError('unknown layout')
    width = int(variant[:2])
    h = np.asarray(row)[-width:]
    if h.ndim != 1 or len(h) != width or np.any(h < 0):
        raise ValueError('full 30/32/34-item histories required')
    if variant == '32_single':
        return [('i', int(i)) for i in h]
    old = width//2
    if variant == '32_reference':
        prefix = [('p', int(h[j]), int(h[j+1])) for j in range(0, old, 2)]
    elif variant == '32_edges':
        prefix = [('i', int(h[0]))] + [('p', int(h[j]), int(h[j+1]))
            for j in range(1, old-1, 2)] + [('i', int(h[old-1]))]
    elif variant.endswith('_first'):
        prefix = [('i', int(h[0]))] + [('p', int(h[j]), int(h[j+1])) for j in range(1, old, 2)]
    else:
        prefix = [('p', int(h[j]), int(h[j+1])) for j in range(0, old-1, 2)] + [('i', int(h[old-1]))]
    return prefix + [('i', int(i)) for i in h[old:]]


def pair_keys(contexts, variant, nitems):
    return np.array(sorted({p[1]*nitems+p[2] for h in contexts
        for p in segments(h, variant) if p[0] == 'p'}), dtype=np.int64)


def encode_layout(contexts, variant, sids, keys, pairs):
    if sids.ndim != 2 or sids.shape[1] != 4 or pairs.shape != (len(keys), 4):
        raise ValueError('every item and pair must retain unique 3+1 SID')
    out = np.empty((len(contexts), POSITIONS[variant]-1), dtype=np.int32)
    for row, h in enumerate(contexts):
        tokens = []
        for part in segments(h, variant):
            if max(part[1:]) >= len(sids):
                raise ValueError('item out of catalogue')
            if part[0] == 'i':
                tokens.extend(sids[part[1]] + np.arange(4)*256)
            else:
                key = part[1]*len(sids)+part[2]
                j = int(np.searchsorted(keys, key))
                if j == len(keys) or keys[j] != key:
                    raise ValueError('pair absent from frozen cache')
                tokens.extend(pairs[j] + (4+np.arange(4))*256)
        if len(tokens) != out.shape[1]:
            raise RuntimeError('layout length differs')
        out[row] = tokens
    return out


def extend_sids(old_keys, old_sids, missing_keys, codes):
    """Append collision ordinals; never renumber any existing 3+1 pair SID."""
    old_keys, missing_keys = map(np.asarray, (old_keys, missing_keys))
    old_sids, codes = map(np.asarray, (old_sids, codes))
    if (old_sids.shape != (len(old_keys), 4) or codes.shape != (len(missing_keys), 3)
        or np.any(old_keys[1:] <= old_keys[:-1]) or np.any(missing_keys[1:] <= missing_keys[:-1])
        or np.intersect1d(old_keys, missing_keys).size or np.any(codes < 0) or np.any(codes >= 256)
        or np.any(old_sids < 0) or np.any(old_sids >= 256)
        or len(np.unique(old_sids, axis=0)) != len(old_sids)):
        raise ValueError('invalid frozen index extension')
    next_suffix = {}
    for sid in old_sids:
        key = tuple(sid[:3]); next_suffix[key] = max(next_suffix.get(key, 0), int(sid[3])+1)
    appended = []
    for code in codes:
        key = tuple(code); suffix = next_suffix.get(key, 0)
        if suffix >= 256:
            raise ValueError('new pair suffix exceeds existing vocabulary')
        appended.append([*code, suffix]); next_suffix[key] = suffix+1
    keys = np.union1d(old_keys, missing_keys)
    values = np.empty((len(keys), 4), dtype=np.int32)
    values[np.searchsorted(keys, old_keys)] = old_sids
    values[np.searchsorted(keys, missing_keys)] = np.asarray(appended, dtype=np.int32).reshape(-1, 4)
    np.testing.assert_array_equal(values[np.searchsorted(keys, old_keys)], old_sids)
    if len(np.unique(values, axis=0)) != len(keys):
        raise RuntimeError('extended pair SIDs collide')
    return keys, values


def prepare_cache(plan_path):
    import torch
    from sentence_transformers import SentenceTransformer
    from rhpr.gr.kindle_pairs import encode_pair_cache
    from rhpr.gr.ml100k_pair_raw import cases
    from rhpr.gr.pair_sid import predict_rq
    p = json.loads(Path(plan_path).read_text()); c = p['config']; validate(c)
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_capability() != (7, 5):
        raise RuntimeError('cache preparation requires isolated T4')
    torch.set_num_threads(1)
    parent = Path(p['parents']['pair'])/'frozen-indexes'
    old_keys = np.load(parent/'pair-keys.npy'); old_sids = np.load(parent/'pairs-42.npy')
    needed = np.unique(np.concatenate([pair_keys(cases(p['data'], s)[0], v, c['data']['items'])
        for s in c['evaluation']['splits'] for v in VARIANTS]))
    missing = np.setdiff1d(needed, old_keys, assume_unique=True)
    if len(missing) != c['expected_new_pairs']:
        raise ValueError('new-pair coverage differs from CPU preflight')
    out = Path(p['cache']); out.mkdir(parents=True, exist_ok=True)
    _event('kindle_alignment_cache_start', needed_pairs=len(needed), new_pairs=len(missing))
    codes = np.empty((0, 3), dtype=np.int32)
    if len(missing):
        texts = json.loads((Path(p['data'])/'texts.json').read_text())
        encoder = SentenceTransformer(str(Path(p['resources'])/'sentence-t5-base'), device='cuda:0', local_files_only=True)
        encoder.max_seq_length = 2048; encoder.eval()
        encode_pair_cache(encoder, texts, missing, out/'new-pair-embeddings.npy', ['First book', 'Second book'])
        del encoder; gc.collect(); torch.cuda.empty_cache()
        codes = predict_rq(np.load(out/'new-pair-embeddings.npy', mmap_mode='r'),
            np.load(parent/'pair-centroids-42.npy'), 'cuda:0')
    keys, sids = extend_sids(old_keys, old_sids, missing, codes)
    np.save(out/'pair-keys.npy', keys); np.save(out/'pairs-42.npy', sids)
    np.save(out/'new-pair-keys.npy', missing)
    report = dict(needed_pairs=len(needed), new_pairs=len(missing), original_pairs=len(old_keys),
        original_sids_unchanged=True, pair_suffix_retained=True, quantizer_refitted=False, training_performed=False)
    _write(out/'cache-report.json', report); _event('kindle_alignment_cache_complete', **report)


def original_prediction(parents, model, split):
    folder = 'mixed' if model == 'mixed32' else 'pair'
    trial = 'mixed32_wide_42' if model == 'mixed32' else 'pair32_wide_42'
    filename = 'validation-epoch-055.npz' if split == 'validation' else 'test-best-predictions.npz'
    return Path(parents[folder])/'trials'/trial/filename


def run_trial(trial, p):
    import torch
    from rhpr.gr.data import file_sha256
    from rhpr.gr.experiment import _loss
    from rhpr.gr.grid import build_grid_generator
    from rhpr.gr.ml100k_pair_raw import cases
    from rhpr.gr.movie_data import case_digest
    from rhpr.gr.pair_unique4 import InputIndex4, attach4
    from rhpr.gr.sid import SidIndex
    from rhpr.gr.sid_study import state_digest
    c = p['config']; name = trial['id']; parent = 'mixed' if name == 'mixed32' else 'pair'
    ident = name+'_wide_42'; checkpoint = Path(p['parents'][parent])/'trials'/ident/'generator-best.pt'
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if state['epoch'] != 55 or state['config']['seed'] != 42 or state['config']['generator'] != c['generator']:
        raise ValueError('checkpoint differs from frozen epoch55 protocol')
    sid = np.load(Path(p['parents']['pair'])/'frozen-indexes/items-42.npy')
    idx = SidIndex.from_full_sids(sid, width=256); wrapped = InputIndex4(idx)
    keys = np.load(Path(p['cache'])/'pair-keys.npy'); pairs = np.load(Path(p['cache'])/'pairs-42.npy')
    net = attach4(build_grid_generator(sid, c['generator'], 42, 'cpu', p['source']))
    net.load_state_dict(state['model'], strict=True); del state
    net.requires_grad_(False); net.eval(); before = state_digest(net.state_dict())
    out = Path(p['output'])/'trials'/name; out.mkdir(parents=True, exist_ok=True)
    # Same B1 math-SDPA supported-forward FLOP definition as the parent studies.
    h, u, t = cases(p['data'], 'validation'); costs = {}
    for v in VARIANTS:
        x = encode_layout(h[:1], v, sid, keys, pairs)
        with torch.inference_mode(), torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH), torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU], with_flops=True) as prof:
            loss = _loss(net, wrapped.model_batch(x, u[:1], t[:1], 'cpu'))
        if not torch.isfinite(loss):
            raise RuntimeError('nonfinite profile loss')
        costs[v] = dict(encoder_positions=x.shape[1]+1, supported_forward_flops_batch1=sum(e.flops for e in prof.key_averages()))
    _write(out/'forward-flops.json', costs)
    net.to('cuda:0'); results = {}
    for split in c['evaluation']['splits']:
        h, u, t = cases(p['data'], split)
        if len(t) != 958 or not np.all(h[:, -34:] >= 0):
            raise ValueError('expected identical 958 full-history targets for every variant')
        results[split] = {}
        for v in VARIANTS:
            x = encode_layout(h, v, sid, keys, pairs)
            metrics, pred, scores = _evaluate(net, (x, u, t), wrapped, 64, 'cuda:0', 'fp32')
            replay = {}
            if v == '32_reference':
                with np.load(original_prediction(p['parents'], name, split)) as z:
                    np.testing.assert_array_equal(u, z['user_ids']); np.testing.assert_array_equal(t, z['target_item_ids'])
                    np.testing.assert_array_equal(pred, z['semantic_ids'])
                    replay = dict(all_ranked_sids_identical=True, max_abs_score_difference=float(np.max(np.abs(scores-z['scores']))))
            if not np.isfinite(scores).all():
                raise RuntimeError('nonfinite prediction scores')
            np.savez_compressed(out/f'{split}-{v}.npz', semantic_ids=pred, scores=scores, user_ids=u, target_item_ids=t)
            results[split][v] = dict(metrics=metrics, case_sha256=case_digest((x, u, t)),
                raw_case_sha256=case_digest((h, u, t)), **costs[v], replay=replay)
            _event('kindle_alignment_evaluation_complete', model=name, split=split, variant=v, metrics=metrics)
    if before != state_digest(net.state_dict()) or any(a.requires_grad or a.grad is not None for a in net.parameters()):
        raise RuntimeError('read-only inference changed model')
    row = dict(trial=name, selected_epoch=55, checkpoint_sha256=file_sha256(checkpoint), loaded_state_sha256=before,
        weights_unchanged=True, training_performed=False, results=results)
    _write(out/'trial-result.json', row); return row


def worker(plan_path, slot):
    import torch, transformers
    p = json.loads(Path(plan_path).read_text()); validate(p['config']); torch.set_num_threads(1)
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_capability() != (7, 5) or transformers.__version__ != '5.22.1':
        raise RuntimeError('requires isolated T4 and HF5.22.1')
    status = dict(results={}, failures=[])
    for trial in p['assignments'][slot]['trials']:
        _event('kindle_alignment_trial_start', trial=trial['id'])
        try:
            row = run_trial(trial, p); status['results'][trial['id']] = row
            _event('kindle_alignment_trial_complete', trial=trial['id'])
        except Exception as error:
            out = Path(p['output'])/'trials'/trial['id']; out.mkdir(parents=True, exist_ok=True)
            (out/'failure.log').write_text(traceback.format_exc())
            status['failures'].append(dict(trial=trial['id'], exception_type=type(error).__name__, message=str(error)))
            _event('kindle_alignment_trial_failed', **status['failures'][-1])
        _write(Path(p['output'])/'workers'/f'worker-{slot}.json', status)
        gc.collect(); torch.cuda.empty_cache()


def run(c, output):
    import torch
    from rhpr.gr.ae_suite import assignments, launch_workers
    from rhpr.gr.data import file_sha256
    from rhpr.gr.grid import prepare_grid_source
    from rhpr.gr.pair_test import locate_parent
    from rhpr.gr.resources import unpack_grid, verify_resources
    from rhpr.gr.study_parallel import worker_environment
    from rhpr.offline_runtime import resolve_dataset_directory
    validate(c); output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=True); (output/'workers').mkdir(exist_ok=True)
    if torch.cuda.device_count() != 2 or any(torch.cuda.get_device_capability(i) != (7, 5) for i in (0, 1)):
        raise RuntimeError('requires two T4s; no fallback')
    torch.set_num_threads(1); parents = {k: locate_parent(v) for k, v in c['parents'].items()}
    for root in parents.values():
        old = json.loads((root/'run-manifest.json').read_text())['config']
        for key in ('generator', 'data', 'movie_input'):
            if c[key] != old[key]:
                raise ValueError('parent differs: '+key)
    for filename in ('items-42.npy', 'pairs-42.npy', 'pair-keys.npy'):
        if file_sha256(parents['pair']/'frozen-indexes'/filename) != file_sha256(parents['mixed']/'frozen-indexes'/filename):
            raise ValueError('frozen indexes differ')
    data = resolve_dataset_directory(c['movie_input']['path'])
    for filename, sha in c['movie_input']['sha256'].items():
        if file_sha256(data/filename) != sha:
            raise ValueError('input differs: '+filename)
    resources = resolve_dataset_directory(c['resources']['path']); verify_resources(resources, c['resources']['manifest_sha256'])
    modern = resolve_dataset_directory(c['modern_runtime']['path']); verify_resources(modern, c['modern_runtime']['manifest_sha256'])
    scratch = Path(c['resources']['scratch_path']); scratch.mkdir(parents=True, exist_ok=True)
    source = prepare_grid_source(unpack_grid(resources/'grid-source.bin', scratch/'original'), scratch/'grid')
    plan = dict(config=c, output=str(output), source=str(source), data=str(data), resources=str(resources),
        cache=str(output/'pair-cache'), parents={k: str(v) for k, v in parents.items()}, assignments=assignments(c, 2))
    path = output/'plan.json'; _write(path, plan)
    subprocess.run([sys.executable, '-u', '-m', 'rhpr.gr.kindle_alignment', '--prepare-cache', str(path)],
        env=worker_environment(c, plan['assignments'][0]), check=True, timeout=900)
    overlay = scratch/'modern-site'; overlay.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps', '--no-cache-dir', '--no-compile', '--require-hashes',
        '--only-binary=:all:', '--target', str(overlay), '--find-links', str(modern), '-r', str(modern/'requirements-lock.txt')]
    with (output/'modern-install.log').open('w') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=180)
    sys.path.insert(0, str(overlay))
    summary = launch_workers(path, worker_module='rhpr.gr.kindle_alignment', event_prefix='kindle_alignment')
    _write(output/'study-summary.json', dict(results=summary['results'], training_performed=False, tokenizer_refitted=False))
    metrics = dict(experiment=c['name'], models=2, evaluations=28, training_performed=False)
    for model, row in summary['results'].items():
        for split, variants in row['results'].items():
            for variant, record in variants.items():
                metrics[f'{model}_{split}_{variant}_ndcg10'] = record['metrics']['ndcg@10']
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--plan'); parser.add_argument('--slot', type=int)
    parser.add_argument('--prepare-cache'); args = parser.parse_args()
    if args.prepare_cache:
        prepare_cache(args.prepare_cache)
    elif args.plan and args.slot is not None:
        worker(args.plan, args.slot)
    else:
        parser.error('specify --prepare-cache or --plan and --slot')
