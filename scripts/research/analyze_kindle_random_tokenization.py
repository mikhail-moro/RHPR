"""CPU-only audit of the completed random-tiling fit and frozen-index controls."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from analyze_kindle_alignment import compare
from rhpr.gr.data import file_sha256
from rhpr.gr.kindle_random_tokenization import POLICY, RandomTokenization, validate
from rhpr.gr.kindle_pairs import encode
from rhpr.gr.ml100k_pair_raw import cases
from rhpr.gr.movie_data import case_digest
from rhpr.gr.sid import SidIndex, evaluate_ranked_sids

SOURCE = 'a7f82485857265d0cb704d54cbb9be892d076519'
CUTOFF = 2.300820444821813


def read(p): return json.loads(Path(p).read_text())
def write(p, x): Path(p).write_text(json.dumps(x, indent=2, allow_nan=False)+'\n')


def analyze(root, parent, mixed, data):
    root, parent, mixed, data = map(Path, (root, parent, mixed, data))
    c = read(root/'experiment.json'); validate(c)
    manifest = read(root/'run-manifest.json'); suite = read(root/'suite-results.json'); study = read(root/'study-summary.json')
    assert manifest['git_sha'] == SOURCE and manifest['config'] == c
    assert suite['results'] == study['results'] and not suite['failures']
    assert study['new_generator_fits'] == 1 and study['resumed_generator_fits'] == 0 and not study['tokenizer_refitted']
    assert study['training_policy'] == POLICY and study['inference_format'] == 'pair32'
    assert (root/'summary.md').is_file()
    assert read(parent/'run-manifest.json')['git_sha'] == c['parent']['git_sha']
    assert read(mixed/'run-manifest.json')['git_sha'] == '88722e9081aae120e9af261f550d50fe74864da6'
    for f, pin in c['parent']['files'].items():
        path = parent/f
        assert path.stat().st_size == pin['size_bytes'] and file_sha256(path) == pin['sha256'], f
        if f.startswith(('trials/', 'baselines/', 'control-tests/')):
            assert file_sha256(root/'baselines'/f) == pin['sha256']
    for f, sha in c['movie_input']['sha256'].items(): assert file_sha256(data/f) == sha
    for f in ('items-42.npy', 'pair-keys.npy', 'pairs-42.npy'):
        assert file_sha256(root/'frozen-indexes'/f) == file_sha256(parent/'frozen-indexes'/f) == file_sha256(mixed/'frozen-indexes'/f)
    raw = {s: cases(data, s) for s in ('train', 'validation', 'test')}
    sid = np.load(root/'frozen-indexes/items-42.npy'); idx = SidIndex.from_full_sids(sid, width=256)
    keys = np.load(root/'frozen-indexes/pair-keys.npy'); pairs = np.load(root/'frozen-indexes/pairs-42.npy')
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
        hit = (pred == sid[raw[split][2], None, :]).all(-1); rank = np.where(hit.any(1), hit.argmax(1)+1, 21)
        credit = np.where(rank <= 10, 1/np.log2(rank+1), 0.)
        assert abs(float(credit.mean())-metric['ndcg@10']) < 1e-12
        files += 1; invalid += metric['invalid_predictions']; duplicates += metric['duplicate_predictions']; paths += metric['prediction_count']
        return rank, credit

    cells = [('random32', root/'trials/random32_wide_42'), ('mixed32', mixed/'trials/mixed32_wide_42'),
        ('pair32', parent/'trials/pair32_wide_42'), ('recent32', parent/'baselines/recent32_wide_42'),
        ('recent20', parent/'baselines/recent20_wide_42')]
    steps = math.ceil(len(raw['train'][2])/512)
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
        if name.startswith('recent'):
            td = parent/'control-tests'/folder.name; test = read(td/'control-result.json')['test']
            testfile = td/'test-best-predictions.npz'
        else: test = row['test']; testfile = folder/'test-best-predictions.npz'
        assert test['epoch'] == selection['best_epoch'] and not test['seen_items_filtered']
        ranks[name, 'test'], credits[name, 'test'] = replay(testfile, 'test', test['metrics'])
        records.append(dict(model=name, best_epoch=selection['best_epoch'], completed_epochs=len(curve), stopping_reason=selection['stopping_reason'],
            validation_ndcg10=selection['validation_ndcg10'], validation_hits10=int((ranks[name, best['epoch']] <= 10).sum()),
            test_ndcg10=test['metrics']['ndcg@10'], test_hits10=int((ranks[name, 'test'] <= 10).sum()),
            test_recall10=test['metrics']['recall@10'], test_ndcg5=test['metrics']['ndcg@5'],
            forward_gflops=row['forward']['supported_forward_flops_batch1']/1e9, parameters=row['forward']['parameters'],
            warm_epoch_seconds=row['timing']['warm_epoch_training_median_seconds'], optimizer_training_seconds=selection['optimizer_training_seconds'],
            min_validation_ce=min(v['validation_loss']['total_ce'] for v in checks),
            min_validation_ce_epoch=min(checks, key=lambda v: v['validation_loss']['total_ce'])['epoch'], amp_skips=selection['amp_skipped_updates']))
    assert len(initial) == 1
    folder = cells[0][1]; row = read(folder/'trial-result.json'); selection = row['selection']
    assert row['random_tokenization'] == POLICY and row['test_case_sha256'] == case_digest(raw['test'])
    assert row['pair_sid_file_sha256'] == file_sha256(root/'frozen-indexes/pairs-42.npy')
    assert study['results'] == {'random32_wide_42': row}
    assert read(root/'metrics.json') == dict(experiment=c['name'], models=1, seed=42, test_evaluated=True,
        validation_ndcg10=selection['validation_ndcg10'], test_ndcg10=row['test']['metrics']['ndcg@10'])
    assert row['forward'] == read(parent/'trials/pair32_wide_42/forward-flops.json')
    for split in ('validation', 'test'):
        h, u, t = raw[split]; x = encode(h, sid, keys, pairs, 32)
        assert x.shape == (958, 96) and np.all(x >= 0) and np.all(x[:, :32] >= 1024) and np.all(x[:, 32:] < 1024)
    provenance = read(folder/'training-view-provenance.json')
    assert provenance['policy'] == POLICY and provenance['target_count'] == len(raw['train'][2]) and not provenance['examples_doubled']
    preflight = read(Path(__file__).resolve().parents[2]/'docs/gr-kindle-random-tokenization-preflight.json')
    assert curves['random32'][0]['training_views'] == preflight['first_epoch']
    from rhpr.gr.kindle_windows import encode_window
    assert provenance['full_case_sha256'] == case_digest((encode_window(raw['train'][0], sid, 32), *raw['train'][1:]))
    assert provenance['pair_case_sha256'] == case_digest((encode(raw['train'][0], sid, keys, pairs, 32), *raw['train'][1:]))
    # Reconstruct every actual sampled input,not merely merge-count expectations.
    views = RandomTokenization(raw['train'][0], sid, keys, pairs)
    for item in curves['random32']:
        ep = item['epoch']; order = np.random.default_rng(42+ep).permutation(len(views.lengths))
        for b, begin in enumerate(range(0, len(order), 512)): views(ep, b, order[begin:begin+512])
        assert views.epoch_summary(ep) == item['training_views'], ep
        if ep % 10 == 0: print(json.dumps(dict(phase='sampled_epoch_inputs_verified', epoch=ep)), flush=True)
    import torch
    torch.set_num_threads(1)
    best_ck = torch.load(folder/'generator-best.pt', map_location='cpu', weights_only=False)
    last_ck = torch.load(folder/'generator-last.pt', map_location='cpu', weights_only=False)
    assert best_ck['epoch'] == selection['best_epoch'] and last_ck['epoch'] == selection['training_epochs']
    assert last_ck['updates'] == selection['optimizer_updates'] and last_ck['scheduler']['last_epoch'] == last_ck['updates']
    assert best_ck['validation']['ndcg@10'] == selection['validation_ndcg10']
    assert file_sha256(folder/'generator-best.pt') == row['test']['checkpoint_sha256']
    for ck in (best_ck, last_ck):
        assert ck['config']['random_tokenization'] == POLICY
        assert ck['config']['generator'] == c['generator'] and ck['config']['seed'] == 42
        assert all(bool(torch.isfinite(v).all()) for v in ck['model'].values() if v.is_floating_point())
    checkpoints = {name: dict(sha256=file_sha256(folder/f'generator-{name}.pt'), size_bytes=(folder/f'generator-{name}.pt').stat().st_size)
        for name in ('best', 'last')}
    del best_ck, last_ck
    by = {v['model']: v for v in records}; comparisons = []; groups = []; recovery = []
    frequency = np.bincount(raw['train'][2], minlength=len(sid))
    for split in ('validation', 'test'):
        h, u, t = raw[split]; low = np.log1p(frequency[h[:, -16:]]).mean(1) <= CUTOFF
        masks = dict(all=np.ones(len(t), dtype=bool), less_popular_recent=low, more_popular_recent=~low,
            target_cold=frequency[t] == 0, target_1_5=(frequency[t] >= 1) & (frequency[t] <= 5),
            target_6_20=(frequency[t] >= 6) & (frequency[t] <= 20), target_over20=frequency[t] > 20)
        user_arrays[split+'_users'] = u; user_arrays[split+'_targets'] = t
        def key(name): return 'test' if split == 'test' else by[name]['best_epoch']
        for name in by:
            credit = credits[name, key(name)]; rank = ranks[name, key(name)]; user_arrays[f'{split}_{name}_rank'] = rank
            for group, mask in masks.items():
                groups.append(dict(split=split, model=name, group=group, users=int(mask.sum()),
                    ndcg10=float(credit[mask].mean()), hits10=int((rank[mask] <= 10).sum())))
            if name == 'random32': continue
            comparisons.append(dict(split=split, comparator=name, **compare(credits['random32', key('random32')], credit,
                ranks['random32', key('random32')], rank)))
        lost = (ranks['recent32', key('recent32')] <= 10) & (ranks['pair32', key('pair32')] > 10)
        recovery.append(dict(split=split, original_pair_lost_recent32_hits=int(lost.sum()),
            recovered_by_random=int((lost & (ranks['random32', key('random32')] <= 10)).sum())))
    late = [dict(epoch=ep, **{name: float(credits[name, ep].mean()) for name in by})
        for ep in sorted(ep for name, ep in credits if name == 'random32' and isinstance(ep, int))]
    for ep in (40, 55, 60):
        for name in by:
            if name == 'random32': continue
            comparisons.append(dict(split=f'validation_fixed{ep}', comparator=name,
                **compare(credits['random32', ep], credits[name, ep], ranks['random32', ep], ranks[name, ep])))
    training = curves['random32']; workerlog = (root/'workers/worker-0.log').read_text()
    fallback_lines = [line for line in workerlog.splitlines() if not line.startswith('{') and
        any(token in line for token in ('hit config.recompile_limit', 'hit config.cache_size_limit'))]
    result = dict(source=SOURCE, session_seconds=manifest['duration_seconds'], completed_fit=True, seed=42,
        completed_epochs=selection['training_epochs'], best_epoch=selection['best_epoch'], capped_at60=selection['stopping_reason'] == 'epoch_limit',
        original_parent_pins_verified=len(c['parent']['files']), input_files_verified=10,
        all_sampled_epoch_inputs_hashes_replayed=len(training), all_cases_sid_initializer_lr_ema_selection_verified=True,
        both_complete_checkpoints_verified=checkpoints, prediction_files_replayed=files, prediction_count=paths,
        invalid_predictions=invalid, duplicate_predictions=duplicates, records=records, comparisons=comparisons, groups=groups,
        validation_by_epoch=late, recovery=recovery, recent_popularity_cutoff=CUTOFF,
        training_pair_occurrences=sum(v['training_views']['pair_summary_occurrences'] for v in training),
        training_odd_start_pair_occurrences=sum(v['training_views']['odd_start_pair_occurrences'] for v in training),
        compile_limit_fallback_lines=fallback_lines,
        train_timing_spikes_over80s=[dict(epoch=v['epoch'], seconds=v['train_seconds']) for v in training if v['train_seconds'] > 80],
        limits=['One examined seed/index/cohort; conditional paired-user CIs omit seed/index/item clustering and model selection uncertainty.',
            'No multiple-comparison correction; TEST already examined,so followup results are exploratory.',
            'Random tiling changes both compression density distribution and pair placements;not an isolated ablation of either.',
            'Historical training timings are not simultaneous benchmarks;forward FLOPs omit beam/backward/offline text encoding.',
            'All five fits capped at60epochs;selected55does not establish convergence.'])
    write(root/'analysis.json', result); np.savez_compressed(root/'user-ranks.npz', **user_arrays)
    with (root/'model-metrics.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    return result, curves


def plot(root, result, curves):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = dict(random32='#bd4440', mixed32='#a145a7', pair32='#278455', recent32='#dd912d', recent20='#3268a8')
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for name, color in colors.items():
        check = [v for v in curves[name] if 'validation' in v]
        axes[0, 0].plot([v['epoch'] for v in check], [v['validation']['ndcg@10'] for v in check], color=color, label=name, marker='.', ms=4)
        axes[0, 1].plot([v['epoch'] for v in check], [v['validation_loss']['total_ce'] for v in check], color=color)
        axes[1, 0].plot([v['epoch'] for v in curves[name]], [v['train_seconds'] for v in curves[name]], color=color)
    for row in result['records']:
        axes[1, 1].scatter(row['forward_gflops'], row['test_ndcg10'], color=colors[row['model']], s=45)
        offset = (-68, 8) if row['model'] == 'random32' else (-62, -15) if row['model'] == 'mixed32' else (5, 5)
        axes[1, 1].annotate(row['model'], (row['forward_gflops'], row['test_ndcg10']), xytext=offset, textcoords='offset points')
    axes[0, 0].set_ylabel('VAL NDCG@10'); axes[0, 0].legend()
    axes[0, 1].set_ylabel('Teacher-forced VAL CE'); axes[1, 0].set_ylabel('Train seconds / epoch incl. cold compile')
    axes[1, 1].set_ylabel('VAL-selected TEST NDCG@10'); axes[1, 1].set_xlabel('B1 supported forward GFLOPs')
    axes[1, 1].margins(x=.25, y=.25)
    for ax in (axes[0, 0], axes[0, 1], axes[1, 0]): ax.set_xlabel('Epoch')
    for ax in axes.flat: ax.grid(alpha=.2)
    fig.savefig(root/'random-curves.png', dpi=180); fig.savefig(root/'random-curves.pdf'); plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('results', 'parent', 'mixed', 'data'): parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--plot', action='store_true'); a = parser.parse_args()
    result, curves = analyze(a.results, a.parent, a.mixed, a.data)
    if a.plot: plot(a.results, result, curves)
    print(json.dumps({k: result[k] for k in ('records', 'comparisons', 'recovery')}, indent=2))
