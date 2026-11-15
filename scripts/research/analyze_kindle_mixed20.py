"""CPU replay of the completed Mixed20/Recent10 fits and frozen controls."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from analyze_kindle_alignment import compare
from rhpr.gr.data import file_sha256
from rhpr.gr.kindle_mixed20 import POLICY, SEEDS, cache_plan, extend_by_split, validate
from rhpr.gr.kindle_pairs import encode
from rhpr.gr.kindle_random_tokenization import RandomTokenization
from rhpr.gr.kindle_windows import encode_window
from rhpr.gr.ml100k_pair_raw import cases
from rhpr.gr.movie_data import case_digest
from rhpr.gr.sid import SidIndex, evaluate_ranked_sids

SOURCE = '4026db0e479881586cad0cdaeae264e0ec4d06ef'
CUTOFF = 2.300820444821813


def read(p): return json.loads(Path(p).read_text())
def write(p, x): Path(p).write_text(json.dumps(x, indent=2, allow_nan=False)+'\n')


def analyze(root, parent, random32, data):
    root, parent, random32, data = map(Path, (root, parent, random32, data))
    c = read(root/'experiment.json'); validate(c)
    manifest = read(root/'run-manifest.json'); study = read(root/'study-summary.json'); suite = read(root/'suite-results.json')
    assert manifest['git_sha'] == SOURCE and manifest['config'] == c
    assert suite['results'] == study['results'] and not suite['failures']
    assert study['new_generator_fits'] == 2 and not study['tokenizer_refitted']
    assert study['seed_roles'] == SEEDS and study['training_policy'] == POLICY
    assert (root/'summary.md').is_file()
    assert read(parent/'run-manifest.json')['git_sha'] == c['parent']['git_sha']
    assert read(random32/'run-manifest.json')['git_sha'] == 'a7f82485857265d0cb704d54cbb9be892d076519'
    for f, pin in c['parent']['files'].items():
        path = parent/f
        assert path.stat().st_size == pin['size_bytes'] and file_sha256(path) == pin['sha256'], f
        if f.startswith(('trials/', 'baselines/', 'control-tests/')):
            assert file_sha256(root/'baselines'/f) == pin['sha256']
    for f, sha in c['movie_input']['sha256'].items(): assert file_sha256(data/f) == sha
    assert file_sha256(root/'frozen-indexes/items-42.npy') == file_sha256(parent/'frozen-indexes/items-42.npy')
    sid = np.load(root/'frozen-indexes/items-42.npy'); idx = SidIndex.from_full_sids(sid, width=256)
    raw = {s: cases(data, s) for s in ('train', 'validation', 'test')}
    keys = np.load(root/'frozen-indexes/pair-keys.npy'); pairs = np.load(root/'frozen-indexes/pairs-42.npy')
    old, sets, missing = cache_plan(data, parent)
    np.testing.assert_array_equal(missing, np.load(root/'pair-cache/new-pair-keys.npy'))
    assert len(missing) == 5011 and len(keys) == 121412
    import torch
    from rhpr.gr.pair_sid import predict_rq
    torch.set_num_threads(1)
    vec = np.load(root/'pair-cache/new-pair-embeddings.npy')
    assert vec.shape == (5011, 768) and vec.dtype == np.float32 and np.isfinite(vec).all()
    codes = predict_rq(vec, np.load(parent/'frozen-indexes/pair-centroids-42.npy'), 'cpu')
    np.testing.assert_array_equal(codes, np.load(root/'pair-cache/new-pair-codes.npy'))
    check_keys, check_pairs = extend_by_split(old, np.load(parent/'frozen-indexes/pairs-42.npy'), sets, missing, codes)
    np.testing.assert_array_equal(keys, check_keys); np.testing.assert_array_equal(pairs, check_pairs)
    cache = read(root/'pair-cache/cache-report.json')
    assert cache['original_sids_unchanged'] and not cache['tokenizer_refitted'] and cache['max_suffix'] == int(pairs[:, 3].max()) == 31
    ranks = {}; credits = {}; curves = {}; records = []; user_arrays = {}; initial = set()
    files = invalid = duplicates = paths = 0

    def replay(file, split, expected):
        nonlocal files, invalid, duplicates, paths
        with np.load(file) as z:
            pred = z['semantic_ids']; scores = z['scores']
            np.testing.assert_array_equal(z['user_ids'], raw[split][1]); np.testing.assert_array_equal(z['target_item_ids'], raw[split][2])
        assert pred.shape == (958, 20, 4) and np.isfinite(scores).all()
        metric = evaluate_ranked_sids(idx, pred, raw[split][2], cutoffs=(5, 10))
        for k, v in metric.items(): assert abs(v-expected[k]) < 1e-12, (file, k)
        hit = (pred == sid[raw[split][2], None, :]).all(-1)
        rank = np.where(hit.any(1), hit.argmax(1)+1, 21)
        credit = np.where(rank <= 10, 1/np.log2(rank+1), 0.)
        assert abs(float(credit.mean())-metric['ndcg@10']) < 1e-12
        files += 1; invalid += metric['invalid_predictions']; duplicates += metric['duplicate_predictions']; paths += metric['prediction_count']
        return rank, credit

    cells = [('mixed20', root/'trials/mixed20_wide_42'), ('recent10', root/'trials/recent10_wide_42'),
        ('recent20', parent/'baselines/recent20_wide_42'), ('recent32', parent/'baselines/recent32_wide_42'),
        ('random32', random32/'trials/random32_wide_42'), ('pair32', parent/'trials/pair32_wide_42')]
    steps = math.ceil(len(raw['train'][2])/512); checkpoints = {}
    for name, folder in cells:
        row = read(folder/'trial-result.json'); selection = row['selection']; curve = read(folder/'training-curves.json'); curves[name] = curve
        assert row['raw_case_sha256'] == {s: case_digest(raw[s]) for s in ('train', 'validation')}
        assert row['case_targets_sha256'] == {s: case_digest(raw[s][1:]) for s in ('train', 'validation')}
        assert row['sid_file_sha256'] == file_sha256(root/'frozen-indexes/items-42.npy')
        initial.add(row['initial_state_sha256']); assert [v['epoch'] for v in curve] == list(range(1, selection['training_epochs']+1))
        previous = 0; ema = reference = last_improvement = None; checks = []
        def lr(update):
            if update < 2*steps: return .003*(update+1)/(2*steps)
            return .001+.002*.5*(1+math.cos(math.pi*min(1., (update-2*steps+1)/(58*steps))))
        for item in curve:
            ep = item['epoch']; assert item['train_examples'] == len(raw['train'][2])
            assert item['updates']+item['amp_skipped_updates'] == steps*ep
            assert abs(item['learning_rate']-lr(previous)) < 1e-14
            assert abs(item['next_learning_rate']-lr(item['updates'])) < 1e-14
            assert min(abs(item['learning_rate_last_batch']-lr(k)) for k in (item['updates']-1, item['updates'])) < 1e-14
            previous = item['updates']
            if 'validation' not in item: continue
            score = item['validation']['ndcg@10']; ema = score if ema is None else .5*ema+.5*score
            if reference is None or ema > reference+.0005: reference = ema; last_improvement = ep
            assert (ema, reference, last_improvement) == (item['validation_ema_ndcg10'], item['ema_reference_ndcg10'], item['ema_last_improvement_epoch'])
            if ep >= 40 and ep < len(curve): assert ep-last_improvement < 15
            ranks[name, ep], credits[name, ep] = replay(folder/f'validation-epoch-{ep:03d}.npz', 'validation', item['validation'])
            checks.append(item)
        best = max(checks, key=lambda v: v['validation']['ndcg@10'])
        assert selection['best_epoch'] == best['epoch'] and selection['validation_ndcg10'] == best['validation']['ndcg@10']
        assert selection['validation_checks'] == len(checks) == len(list(folder.glob('validation-epoch-*.npz')))
        assert selection['optimizer_updates'] == curve[-1]['updates'] and selection['amp_skipped_updates'] == curve[-1]['amp_skipped_updates']
        assert selection['final_early_stopping_state'] == dict(ema=ema, reference=reference, last_improvement_epoch=last_improvement)
        assert abs(selection['optimizer_training_seconds']-sum(v['train_seconds'] for v in curve)) < 1e-6
        if selection['stopping_reason'] == 'epoch_limit': assert len(curve) == 60
        else: assert selection['stopping_reason'] == 'early_stopping' and len(curve) >= 40 and len(curve)-last_improvement >= 15
        if name in ('recent20', 'recent32'):
            td = parent/'control-tests'/folder.name; test = read(td/'control-result.json')['test']; testfile = td/'test-best-predictions.npz'
        else: test = row['test']; testfile = folder/'test-best-predictions.npz'
        assert test['epoch'] == selection['best_epoch'] and not test['seen_items_filtered']
        ranks[name, 'test'], credits[name, 'test'] = replay(testfile, 'test', test['metrics'])
        records.append(dict(model=name, best_epoch=selection['best_epoch'], completed_epochs=len(curve), stopping_reason=selection['stopping_reason'],
            validation_ndcg10=selection['validation_ndcg10'], validation_hits10=int((ranks[name, best['epoch']] <= 10).sum()),
            test_ndcg10=test['metrics']['ndcg@10'], test_hits10=int((ranks[name, 'test'] <= 10).sum()),
            test_hits1=int((ranks[name, 'test'] == 1).sum()), test_ndcg5=test['metrics']['ndcg@5'],
            forward_gflops=row['forward']['supported_forward_flops_batch1']/1e9, parameters=row['forward']['parameters'],
            warm_epoch_seconds=row['timing']['warm_epoch_training_median_seconds'], optimizer_training_seconds=selection['optimizer_training_seconds'],
            min_validation_ce=min(v['validation_loss']['total_ce'] for v in checks),
            min_validation_ce_epoch=min(checks, key=lambda v: v['validation_loss']['total_ce'])['epoch'], amp_skips=selection['amp_skipped_updates']))
        if name not in ('mixed20', 'recent10'): continue
        assert study['results'][row['trial']] == row and row['seed_roles'] == SEEDS
        assert row['pair_sid_file_sha256'] == file_sha256(root/'frozen-indexes/pairs-42.npy')
        provenance = read(folder/'training-view-provenance.json')
        for split in ('train', 'validation'):
            h, u, t = raw[split]
            x = encode(h, sid, keys, pairs, 20) if name == 'mixed20' else encode_window(h, sid, 10)
            assert provenance['evaluation_case_sha256'][split] == case_digest((x, u, t))
            if split == 'validation': assert x.shape == (958, 60 if name == 'mixed20' else 40) and np.all(x >= 0)
        checkpoints[name] = {}
        for kind, epoch in [('best', selection['best_epoch']), ('last', len(curve))]:
            file = folder/f'generator-{kind}.pt'; ck = torch.load(file, map_location='cpu', weights_only=False)
            assert ck['epoch'] == epoch and ck['config']['seed_roles'] == SEEDS
            assert ck['config']['generator'] == c['generator'] and ck['config']['seed'] == 42
            assert all(bool(torch.isfinite(v).all()) for v in ck['model'].values() if v.is_floating_point())
            if kind == 'last': assert ck['updates'] == selection['optimizer_updates'] and ck['scheduler']['last_epoch'] == ck['updates']
            if kind == 'best': assert file_sha256(file) == test['checkpoint_sha256'] and ck['validation']['ndcg@10'] == selection['validation_ndcg10']
            checkpoints[name][kind] = dict(epoch=epoch, sha256=file_sha256(file), size_bytes=file.stat().st_size)
            del ck
    assert len(initial) == 1
    # Replay ALL actually sampled TRAIN tensors,not just count expectations.
    views = RandomTokenization(raw['train'][0], sid, keys, pairs, history_items=20, augmentation_seed=42)
    for item in curves['mixed20']:
        ep = item['epoch']; order = np.random.default_rng(42+ep).permutation(len(views.lengths))
        for b, begin in enumerate(range(0, len(order), 512)): views(ep, b, order[begin:begin+512])
        assert views.epoch_summary(ep) == item['training_views'], ep
        if ep % 10 == 0: print(json.dumps(dict(phase='sampled_epoch_inputs_verified', epoch=ep)), flush=True)
    by = {v['model']: v for v in records}; comparisons = []; groups = []; late = []
    frequency = np.bincount(raw['train'][2], minlength=len(sid))
    pairs_to_compare = [('mixed20', name) for name in ('recent10', 'recent20', 'recent32', 'random32', 'pair32')]+[('recent10', 'recent20')]
    for split in ('validation', 'test'):
        h, u, t = raw[split]; low = np.log1p(frequency[h[:, -16:]]).mean(1) <= CUTOFF
        masks = dict(all=np.ones(len(t), dtype=bool), less_popular_recent=low, more_popular_recent=~low,
            target_cold=frequency[t] == 0, target_1_5=(frequency[t] >= 1)&(frequency[t] <= 5),
            target_6_20=(frequency[t] >= 6)&(frequency[t] <= 20), target_over20=frequency[t] > 20)
        user_arrays[split+'_users'] = u; user_arrays[split+'_targets'] = t
        def key(name): return 'test' if split == 'test' else by[name]['best_epoch']
        for name in by:
            rank = ranks[name, key(name)]; credit = credits[name, key(name)]; user_arrays[f'{split}_{name}_rank'] = rank
            for group, mask in masks.items():
                groups.append(dict(split=split, model=name, group=group, users=int(mask.sum()),
                    ndcg10=float(credit[mask].mean()), hits10=int((rank[mask] <= 10).sum())))
        for left, right in pairs_to_compare:
            comparisons.append(dict(split=split, left=left, right=right,
                **compare(credits[left, key(left)], credits[right, key(right)], ranks[left, key(left)], ranks[right, key(right)])))
    for ep in (40, 45, 50, 55, 60):
        late.append(dict(epoch=ep, **{name: float(credits[name, ep].mean()) for name in by}))
    worker_logs = '\n'.join(p.read_text() for p in (root/'workers').glob('*.log'))
    fallbacks = [line for line in worker_logs.splitlines() if not line.startswith('{') and
        any(token in line for token in ('hit config.recompile_limit', 'hit config.cache_size_limit'))]
    metrics = dict(experiment=c['name'], models=2, seed=42, test_evaluated=True)
    for name in ('mixed20', 'recent10'):
        ident = name+'_wide_42'; row = by[name]
        metrics[ident+'_validation_ndcg10'] = row['validation_ndcg10']; metrics[ident+'_test_ndcg10'] = row['test_ndcg10']
        metrics[ident+'_forward_flops'] = int(round(row['forward_gflops']*1e9))
    assert read(root/'metrics.json') == metrics
    result = dict(source=SOURCE, session_seconds=manifest['duration_seconds'], completed_fits=2, seed_roles=SEEDS,
        original_parent_pins_verified=len(c['parent']['files']), input_files_verified=10, new_pair_vectors_and_codes_replayed=len(missing),
        original_pair_sids_unchanged=True, index_extension=cache, all_sampled_epoch_input_hashes_replayed=len(curves['mixed20']),
        all_cases_initializer_lr_ema_selection_verified=True, checkpoints=checkpoints, records=records,
        prediction_files_replayed=files, prediction_count=paths, invalid_predictions=invalid, duplicate_predictions=duplicates,
        comparisons=comparisons, groups=groups, validation_late=late, compile_limit_fallback_lines=fallbacks,
        training_pair_occurrences=sum(v['training_views']['pair_summary_occurrences'] for v in curves['mixed20']),
        training_odd_start_pair_occurrences=sum(v['training_views']['odd_start_pair_occurrences'] for v in curves['mixed20']),
        limits=['One examined seed/index/cohort; conditional paired-user CIs omit seed/index/item clustering and model selection uncertainty.',
            'No multiple-comparison correction; TEST previously examined,followup exploratory.',
            'Both new fits select the60epoch cap,not established convergence;controls select55.',
            'Recent10is a cheaper recent-only control,not a FLOP-matched control. Historical timings are not simultaneous benchmarks.',
            'Forward FLOPs exclude beam/backward/offline text encoding;not serving latency.'])
    write(root/'analysis.json', result); np.savez_compressed(root/'user-ranks.npz', **user_arrays)
    with (root/'model-metrics.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    return result, curves


def plot(root, result, curves):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = dict(mixed20='#b44543', recent10='#248f90', recent20='#3268a8', recent32='#dd912d', random32='#9652a0', pair32='#278455')
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for name, color in colors.items():
        checks = [v for v in curves[name] if 'validation' in v]
        axes[0, 0].plot([v['epoch'] for v in checks], [v['validation']['ndcg@10'] for v in checks], color=color, label=name, marker='.', ms=4)
        axes[0, 1].plot([v['epoch'] for v in checks], [v['validation_loss']['total_ce'] for v in checks], color=color)
    for ax, field, label in [(axes[1, 0], 'validation_ndcg10', 'VAL NDCG@10'), (axes[1, 1], 'test_ndcg10', 'VAL-selected TEST NDCG@10')]:
        for row in result['records']:
            name = row['model']; ax.scatter(row['forward_gflops'], row[field], color=colors[name], s=45)
            offset = (5, -12) if name in ('recent10', 'random32') else (5, 5)
            ax.annotate(name, (row['forward_gflops'], row[field]), xytext=offset, textcoords='offset points', fontsize=8)
        ax.set_ylabel(label); ax.set_xlabel('B1 supported forward GFLOPs'); ax.margins(x=.18, y=.2)
    axes[0, 0].set_ylabel('VAL NDCG@10'); axes[0, 0].legend(fontsize=8, ncol=2)
    axes[0, 1].set_ylabel('Teacher-forced VAL CE')
    for ax in axes[0]: ax.set_xlabel('Epoch')
    for ax in axes.flat: ax.grid(alpha=.2)
    fig.savefig(root/'mixed20-curves.png', dpi=180); fig.savefig(root/'mixed20-curves.pdf'); plt.close(fig)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('results', 'parent', 'random32', 'data'): p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args(); result, curves = analyze(a.results, a.parent, a.random32, a.data); plot(a.results, result, curves)
    print(json.dumps({k: result[k] for k in ('records', 'comparisons')}, indent=2))
