"""Independent CPU-only replay of frozen alignment/full-single evaluations."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from rhpr.gr.data import file_sha256
from rhpr.gr.kindle_alignment import VARIANTS, encode_layout, extend_sids, original_prediction, pair_keys, segments, validate
from rhpr.gr.movie_data import case_digest
from rhpr.gr.sid import SidIndex, evaluate_ranked_sids

SOURCE = 'a2de37f0d0a30fe4953b5e42b41b174c0c7e6902'
CUTOFF = 2.300820444821813


def read(path): return json.loads(Path(path).read_text())
def write(path, value): Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def compare(left, right, left_rank, right_rank):
    delta = left-right; rng = np.random.default_rng(4242); means = []
    for _ in range(10):
        boot = rng.integers(0, len(delta), size=(500, len(delta)))
        means.extend(delta[boot].mean(1))
    gained = (left_rank <= 10) & (right_rank > 10)
    lost = (left_rank > 10) & (right_rank <= 10)
    shared = (left_rank <= 10) & (right_rank <= 10)
    return dict(delta_ndcg10=float(delta.mean()), conditional_user_ci95=np.quantile(means, [.025, .975]).tolist(),
        improved_users=int((delta > 0).sum()), worsened_users=int((delta < 0).sum()),
        gained_hits=int(gained.sum()), lost_hits=int(lost.sum()), shared_hits=int(shared.sum()),
        gain_from_new_hits=float(delta[gained].sum()/len(delta)),
        loss_from_lost_hits=float(delta[lost].sum()/len(delta)), shared_rank_change=float(delta[shared].sum()/len(delta)))


def analyze(root, pair, mixed, data):
    root, pair, mixed, data = map(Path, (root, pair, mixed, data))
    c = read(root/'experiment.json'); validate(c)
    manifest = read(root/'run-manifest.json'); suite = read(root/'suite-results.json'); study = read(root/'study-summary.json')
    assert manifest['git_sha'] == SOURCE and manifest['config'] == c
    assert suite['results'] == study['results'] and not suite['failures']
    assert study['training_performed'] is False and study['tokenizer_refitted'] is False
    assert (root/'summary.md').is_file()
    parents = dict(pair=pair, mixed=mixed); pins = 0
    for name, spec in c['parents'].items():
        assert read(parents[name]/'run-manifest.json')['git_sha'] == spec['git_sha']
        for filename, pin in spec['files'].items():
            path = parents[name]/filename
            assert path.stat().st_size == pin['size_bytes'] and file_sha256(path) == pin['sha256'], filename
            pins += 1
    for filename, sha in c['movie_input']['sha256'].items(): assert file_sha256(data/filename) == sha
    raw = {}
    for split in ('train', 'validation', 'test'):
        with np.load(data/f'{split}-cases.npz') as z: raw[split] = tuple(z[k] for k in ('contexts', 'user_ids', 'targets'))
    np.testing.assert_array_equal(raw['validation'][1], raw['test'][1])
    np.testing.assert_array_equal(raw['test'][0][:, :-1], raw['validation'][0][:, 1:])
    np.testing.assert_array_equal(raw['test'][0][:, -1], raw['validation'][2])
    sid = np.load(pair/'frozen-indexes/items-42.npy'); idx = SidIndex.from_full_sids(sid, width=256)
    old_keys = np.load(pair/'frozen-indexes/pair-keys.npy'); old_sids = np.load(pair/'frozen-indexes/pairs-42.npy')
    keys = np.load(root/'pair-cache/pair-keys.npy'); pairs = np.load(root/'pair-cache/pairs-42.npy')
    missing = np.load(root/'pair-cache/new-pair-keys.npy'); assert len(missing) == 838
    np.testing.assert_array_equal(pairs[np.searchsorted(keys, old_keys)], old_sids)
    assert len(np.unique(pairs, axis=0)) == len(keys) and pairs.shape == (len(keys), 4)
    assert np.all(pairs >= 0) and np.all(pairs < 256)
    needed = np.unique(np.concatenate([pair_keys(raw[s][0], v, len(sid)) for s in ('validation', 'test') for v in VARIANTS]))
    np.testing.assert_array_equal(missing, np.setdiff1d(needed, old_keys))
    np.testing.assert_array_equal(keys, np.union1d(old_keys, missing))
    report = read(root/'pair-cache/cache-report.json')
    assert report == dict(needed_pairs=len(needed), new_pairs=838, original_pairs=len(old_keys),
        original_sids_unchanged=True, pair_suffix_retained=True, quantizer_refitted=False, training_performed=False)
    import torch
    from rhpr.gr.pair_sid import predict_rq
    torch.set_num_threads(1)
    vec = np.load(root/'pair-cache/new-pair-embeddings.npy')
    assert vec.shape == (838, 768) and vec.dtype == np.float32 and np.isfinite(vec).all()
    codes = predict_rq(vec, np.load(pair/'frozen-indexes/pair-centroids-42.npy'), 'cpu')
    check_keys, check_sids = extend_sids(old_keys, old_sids, missing, codes)
    np.testing.assert_array_equal(keys, check_keys); np.testing.assert_array_equal(pairs, check_sids)
    frequency = np.bincount(raw['train'][2], minlength=len(sid))
    training_pairs = pair_keys_for_train(raw['train'][0], len(sid))
    # Every original TRAIN example actually received Pair input at least once
    # over60Mixed epochs; this checks occurrence novelty,not just nominal input.
    from rhpr.gr.kindle_mixed_views import MixedViews
    n = len(raw['train'][2]); views = MixedViews(np.broadcast_to(0, (n, 128)), np.broadcast_to(0, (n, 96)))
    seen = np.zeros(n, dtype=bool)
    for epoch in range(1, 61):
        order = np.random.default_rng(42+epoch).permutation(n)
        schedule = views.schedule(epoch)
        for b, begin in enumerate(range(0, n, 512)):
            if not schedule[b]: seen[order[begin:begin+512]] = True
    assert seen.all()
    records = []; groups = []; comparisons = []; ranks = {}; credits = {}; user_arrays = {}; coverage = {}
    metrics = read(root/'metrics.json'); assert metrics['models'] == 2 and metrics['evaluations'] == 28 and metrics['training_performed'] is False
    invalid = duplicates = count = 0
    preflight = read(Path(__file__).resolve().parents[2]/'docs/gr-kindle-alignment-preflight.json')
    for split in ('validation', 'test'):
        h, u, t = raw[split]; assert len(t) == 958 and np.all(h[:, -34:] >= 0)
        meanpop = np.log1p(frequency[h[:, -16:]]).mean(1); low = meanpop <= CUTOFF
        masks = dict(all=np.ones(len(t), dtype=bool), less_popular_recent=low, more_popular_recent=~low,
            target_cold=frequency[t] == 0, target_1_5=(frequency[t] >= 1) & (frequency[t] <= 5),
            target_6_20=(frequency[t] >= 6) & (frequency[t] <= 20), target_over20=frequency[t] > 20)
        user_arrays[split+'_users'] = u; user_arrays[split+'_targets'] = t
        for v in VARIANTS:
            queries = [[p[1]*len(sid)+p[2] for p in segments(row, v) if p[0] == 'p'] for row in h]
            q = np.asarray(queries, dtype=np.int64); novel = ~np.isin(q, training_pairs)
            coverage[split+'_'+v] = dict(pair_occurrences=int(q.size), novel_training_pair_occurrences=int(novel.sum()),
                users_with_novel_pair=int(novel.any(1).sum()), new_cache_pair_occurrences=int(np.isin(q, missing).sum()))
        for name in ('mixed32', 'pair32'):
            row = study['results'][name]; assert row == read(root/'trials'/name/'trial-result.json')
            parent = 'mixed' if name == 'mixed32' else 'pair'
            cp = f'trials/{name}_wide_42/generator-best.pt'
            assert row['checkpoint_sha256'] == c['parents'][parent]['files'][cp]['sha256']
            assert row['selected_epoch'] == 55 and row['weights_unchanged'] and row['training_performed'] is False
            assert row['loaded_state_sha256'] == preflight['native_checkpoint_replay'][name]['loaded_state_sha256']
            for v in VARIANTS:
                record = row['results'][split][v]; x = encode_layout(h, v, sid, keys, pairs)
                assert record['case_sha256'] == case_digest((x, u, t)) and record['raw_case_sha256'] == case_digest((h, u, t))
                assert record['encoder_positions'] == x.shape[1]+1
                assert record['supported_forward_flops_batch1'] == preflight['costs'][name][v]['supported_forward_flops_batch1']
                with np.load(root/'trials'/name/f'{split}-{v}.npz') as z:
                    pred = z['semantic_ids']; scores = z['scores']
                    np.testing.assert_array_equal(u, z['user_ids']); np.testing.assert_array_equal(t, z['target_item_ids'])
                assert pred.shape == (958, 20, 4) and np.isfinite(scores).all()
                metric = evaluate_ranked_sids(idx, pred, t, cutoffs=(5, 10))
                for k, value in metric.items(): assert abs(value-record['metrics'][k]) < 1e-12
                assert abs(metric['ndcg@10']-metrics[f'{name}_{split}_{v}_ndcg10']) < 1e-12
                hit = (pred == sid[t, None, :]).all(-1); rank = np.where(hit.any(1), hit.argmax(1)+1, 21)
                credit = np.where(rank <= 10, 1/np.log2(rank+1), 0.)
                assert abs(float(credit.mean())-metric['ndcg@10']) < 1e-12
                if v == '32_reference':
                    with np.load(original_prediction(parents, name, split)) as z:
                        np.testing.assert_array_equal(pred, z['semantic_ids']); np.testing.assert_array_equal(scores, z['scores'])
                    assert record['replay'] == dict(all_ranked_sids_identical=True, max_abs_score_difference=0.)
                ranks[split, name, v] = rank; credits[split, name, v] = credit
                user_arrays[f'{split}_{name}_{v}_rank'] = rank
                records.append(dict(split=split, model=name, variant=v, ndcg10=metric['ndcg@10'], recall10=metric['recall@10'],
                    hits10=int((rank <= 10).sum()), encoder_positions=x.shape[1]+1, forward_gflops=record['supported_forward_flops_batch1']/1e9))
                for group, mask in masks.items():
                    groups.append(dict(split=split, model=name, variant=v, group=group, users=int(mask.sum()),
                        ndcg10=float(credit[mask].mean()), hits10=int((rank[mask] <= 10).sum())))
                invalid += metric['invalid_predictions']; duplicates += metric['duplicate_predictions']; count += 1
        for name in ('mixed32', 'pair32'):
            for v in VARIANTS[1:]:
                a = split, name, v; b = split, name, '32_reference'
                comparisons.append(dict(split=split, model=name, variant=v, reference='32_reference',
                    **compare(credits[a], credits[b], ranks[a], ranks[b])))
        for v in VARIANTS:
            a = split, 'mixed32', v; b = split, 'pair32', v
            comparisons.append(dict(split=split, model='mixed_minus_pair', variant=v, reference='same_layout',
                **compare(credits[a], credits[b], ranks[a], ranks[b])))
    # Reuse the already evaluated Recent32 control,without additional inference.
    control = []; control_pins = 0
    source_pins = read(mixed/'experiment.json')['parent']['files']
    for split in ('validation', 'test'):
        filename = ('baselines/recent32_wide_42/validation-epoch-055.npz' if split == 'validation'
            else 'control-tests/recent32_wide_42/test-best-predictions.npz')
        path = pair/filename; pin = source_pins[filename]
        assert file_sha256(path) == pin['sha256'] and path.stat().st_size == pin['size_bytes']; control_pins += 1
        with np.load(path) as z:
            pred = z['semantic_ids']; np.testing.assert_array_equal(z['user_ids'], raw[split][1])
            np.testing.assert_array_equal(z['target_item_ids'], raw[split][2])
        metric = evaluate_ranked_sids(idx, pred, raw[split][2], cutoffs=(5, 10))
        hit = (pred == sid[raw[split][2], None, :]).all(-1); rank = np.where(hit.any(1), hit.argmax(1)+1, 21)
        credit = np.where(rank <= 10, 1/np.log2(rank+1), 0.)
        control.append(dict(split=split, model='recent32', ndcg10=metric['ndcg@10'], hits10=int((rank <= 10).sum()),
            forward_gflops=.704769968, reused_saved_predictions=True))
        a = split, 'mixed32', '32_single'
        comparisons.append(dict(split=split, model='mixed_full_minus_recent32', variant='32_single', reference='recent32_saved',
            **compare(credits[a], credit, ranks[a], rank)))
    overlap = []
    for v in VARIANTS:
        counts = []
        for vh, th in zip(raw['validation'][0], raw['test'][0], strict=True):
            left = {p[1:] for p in segments(vh, '32_reference') if p[0] == 'p'}
            right = {p[1:] for p in segments(th, v) if p[0] == 'p'}
            counts.append(len(left & right))
        overlap.append(dict(validation_layout='32_reference', test_layout=v, minimum_shared_pairs=min(counts),
            maximum_shared_pairs=max(counts), mean_shared_pairs=float(np.mean(counts))))
    np.savez_compressed(root/'user-ranks.npz', **user_arrays)
    with (root/'model-metrics.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    result = dict(source=SOURCE, session_seconds=manifest['duration_seconds'], parent_pins_verified=pins, input_files_verified=10,
        training_performed=False, prediction_files_replayed=count, prediction_paths=count*958*20,
        invalid_predictions=invalid, duplicate_predictions=duplicates, all_reference_predictions_and_scores_bit_identical=True,
        all_new_codes_and_suffixes_reproduced_on_cpu=True, actual_mixed_train_examples_ever_assigned_pair_view=int(seen.sum()),
        all_original_train_pair_keys_presented_in_mixed=True, additional_saved_control_pins_verified=control_pins,
        records=records, groups=groups, comparisons=comparisons, pair_coverage=coverage, val_reference_test_pair_overlap=overlap,
        reused_recent32_control=control,
        limits=['One already inspected seed and TEST; no new training or independent generalization evidence.',
            'Conditional paired-user CIs exclude seed/index/checkpoint-selection uncertainty and multiplicity correction.',
            '30/34 layouts change context and recent-tail length; boundary variants change token positions and counts.',
            'Full32 is trained format for Mixed and OOD for Pair-only; changing inference does not estimate retraining.'])
    assert count == 28
    write(root/'analysis.json', result); return result


def pair_keys_for_train(contexts, nitems):
    from rhpr.gr.kindle_pairs import collect_keys
    return collect_keys(contexts, 32, nitems)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('results', 'pair', 'mixed', 'data'):
        parser.add_argument('--'+name, type=Path, required=True)
    a = parser.parse_args(); r = analyze(a.results, a.pair, a.mixed, a.data)
    print(json.dumps({k: r[k] for k in ('session_seconds', 'prediction_files_replayed', 'invalid_predictions', 'duplicate_predictions')}, indent=2))
