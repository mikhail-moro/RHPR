"""Read-only CPU verification and calibration of a partially completed mini suite."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from rhpr.gr.mini_tiger import ARCHITECTURES, rank_correlation, selection_stats, validate
from rhpr.gr.movie_data import case_digest
from rhpr.gr.sid import SidIndex


def load(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def replay(root, data, out):
    out.mkdir(parents=True, exist_ok=True)
    c = load(root / 'experiment.json'); validate(c)
    assert c == load(root / 'plan.json')['config']
    suite = load(root / 'suite-results.json'); rows = suite['results']
    assert set(rows) <= {r['id'] for r in c['trials']}
    assert suite['failures'] and all(r['exception_type'] == 'TimeoutError' for r in suite['failures'])
    workers = [load(p) for p in sorted((root / 'workers').glob('worker-*.json'))]
    assert len(workers) == 2
    assert {k: v for w in workers for k, v in w['results'].items()} == rows
    assert sum(len(w['failures']) for w in workers) == len(suite['failures'])
    with np.load(data / 'validation-cases.npz', allow_pickle=False) as z:
        cases = tuple(z[k] for k in ['contexts', 'user_ids', 'targets'])
    assert case_digest(cases) == c['data']['case_sha256']['validation']
    assert len(cases[1]) == 2786
    for split in ['train']:
        with np.load(data / (split + '-cases.npz'), allow_pickle=False) as z:
            assert case_digest(tuple(z[k] for k in ['contexts', 'user_ids', 'targets'])) == c['data']['case_sha256'][split]
    refs = load(root / 'frozen-indexes/references.json')
    checks = []; records = []; scores = {}; curves = {}; full_scores = {}
    for i in c['index_ids']:
        seed = str(refs[i]['full_seed'])
        for filename, digest in c['parents'][seed]['artifacts'][i].items():
            assert sha(root / 'frozen-indexes' / (i + '-' + filename)) == digest
        sid = np.load(root / 'frozen-indexes' / (i + '-semantic-ids.npy'), allow_pickle=False)
        assert sid.shape == (2874, 4); SidIndex.from_full_sids(sid)
        full_curves = load(root / 'frozen-indexes' / (i + '-training-curves.json'))
        full_scores[i] = {str(r['epoch']): r['validation'] for r in full_curves if 'validation' in r}
        assert refs[i]['epochs'] == {e: r['ndcg@10'] for e, r in full_scores[i].items()}
    for name, row in rows.items():
        trial = next(r for r in c['trials'] if r['id'] == name)
        assert all(row[k] == trial[k] for k in ['architecture', 'index_id', 'seed'])
        p = root / 'trials' / name
        assert load(p / 'trial-result.json') == row
        gen = load(p / 'generator.json'); sel = row['selection']; curve = load(p / 'training-curves.json')
        assert gen['initial_state_sha256'] == c['initialization_pins'][trial['architecture']][str(trial['seed'])]
        assert gen['config'] == {**c['generator'], **ARCHITECTURES[trial['architecture']]}
        assert gen['sids_sha256'] == refs[trial['index_id']]['sids_sha256']
        assert sel['training_epochs'] == 20 and sel['optimizer_updates'] == 5920
        assert sel['training_examples_per_epoch'] == 151166 and sel['validation_checks'] == 4
        assert sel['amp_skipped_updates'] == 0 and len(curve) == 20
        assert all(r['updates'] == 296 * r['epoch'] and r['train_examples'] == 151166 for r in curve)
        assert sel['training_precision'] == 'fp16' and sel['evaluation_precision'] == 'fp32'
        assert [r['epoch'] for r in curve if 'validation' in r] == [5, 10, 15, 20]
        best = max((r for r in curve if 'validation' in r), key=lambda r: r['validation']['ndcg@10'])
        assert sel['best_epoch'] == best['epoch'] and sel['validation_ndcg10'] == best['validation']['ndcg@10']
        sid = np.load(root / 'frozen-indexes' / (trial['index_id'] + '-semantic-ids.npy'), allow_pickle=False)
        pack = np.asarray([256**3, 256**2, 256, 1], dtype=np.int64)
        catalogue = np.sort(sid @ pack)
        curves[name] = curve
        for r in curve:
            if 'validation' not in r: continue
            epoch = str(r['epoch']); expected = r['validation']
            path = p / f"validation-epoch-{r['epoch']:03d}.npz"
            with np.load(path, allow_pickle=False) as z:
                np.testing.assert_array_equal(z['user_ids'], cases[1]); np.testing.assert_array_equal(z['target_item_ids'], cases[2])
                pred = z['semantic_ids']; assert pred.shape == (2786, 20, 4)
                assert np.isfinite(z['scores']).all() and np.all((pred >= 0) & (pred < 256))
                hit = (pred == sid[cases[2], None, :]).all(-1)
                rank = np.where(hit.any(1), hit.argmax(1) + 1, 0)
                keys = pred @ pack; pos = np.searchsorted(catalogue, keys)
                valid = (pos < len(catalogue)) & (catalogue[np.minimum(pos, len(catalogue) - 1)] == keys)
                invalid = int((~valid).sum())
                dup = sum(len(k[v]) - len(np.unique(k[v])) for k, v in zip(keys, valid))
            assert invalid == expected['invalid_predictions'] and dup == expected['duplicate_predictions']
            values = {}
            for k in [5, 10]:
                ok = (rank > 0) & (rank <= k)
                values[f'ndcg@{k}'] = np.where(ok, 1 / np.log2(np.maximum(rank, 1) + 1), 0.)
                values[f'recall@{k}'] = ok.astype(float)
            for k, v in values.items(): np.testing.assert_allclose(v.mean(), expected[k], atol=1e-12, rtol=0)
            assert row['epochs'][epoch] == expected['ndcg@10']
            scores[name, epoch] = values['ndcg@10']
            checks.append(dict(path=str(path.relative_to(root)), sha256=sha(path), invalid=invalid, duplicates=int(dup)))
        records.append(dict(**trial, parameters=gen['parameters'],
            epoch_seconds=sel['optimizer_training_seconds'] / 20, wall_seconds=sel['training_seconds'],
            amp_skips=sel['amp_skipped_updates'], final_val=row['epochs']['20']))
    complete = [a for a in ARCHITECTURES if sum(r['architecture'] == a for r in rows.values()) == 16]
    calibration = {}; sensitivity = {}
    for group, ids in [('all', c['index_ids'])] + [(f'index_seed{s}', [i for i in c['index_ids'] if refs[i]['full_seed'] == s]) for s in [42, 101]]:
        report = {}; target = [refs[i]['epochs']['30'] for i in ids]
        for a in complete:
            for e in ['5', '10', '15', '20']:
                names = {s: [next(k for k, r in rows.items() if (r['architecture'], r['index_id'], r['seed']) == (a, i, s)) for i in ids] for s in c['generator_seeds']}
                vals = {s: [rows[n]['epochs'][e] for n in ns] for s, ns in names.items()}
                mean = np.mean(list(vals.values()), axis=0).tolist()
                selected_target = [max(refs[i]['epochs'].values()) for i in ids]
                report[f'{a}_{e}'] = dict(mean=selection_stats(mean, target),
                    seeds={str(s): selection_stats(v, target) for s, v in vals.items()},
                    scores=vals, mean_scores=mean, selected_index=ids[int(np.argmax(mean))],
                    secondary_full_selected=selection_stats(mean, selected_target))
                if group == 'all':
                    loo = [rank_correlation([v for j, v in enumerate(mean) if j != drop], [v for j, v in enumerate(target) if j != drop]) for drop in range(8)]
                    centered = [np.asarray(v, dtype=float).copy() for v in [mean, target]]
                    for s in [42, 101]:
                        ix = [j for j, i in enumerate(ids) if refs[i]['full_seed'] == s]
                        for v in centered: v[ix] -= np.mean(v[ix])
                    sensitivity[f'{a}_{e}'] = dict(leave_one_index_out_rho=loo, within_group_centered_rho=rank_correlation(*centered))
        for e in ['5', '10', '15', '20']:
            report[f'full_{e}'] = selection_stats([refs[i]['epochs'][e] for i in ids], target)
        calibration[group] = dict(index_ids=ids, target=target, comparisons=report)
    timing = {}
    for a in ARCHITECTURES:
        rs = [r for r in records if r['architecture'] == a]
        timing[a] = dict(fits=len(rs), mean_epoch_seconds=float(np.mean([r['epoch_seconds'] for r in rs])),
                         mean_fit_wall_seconds=float(np.mean([r['wall_seconds'] for r in rs])))
    timing['full'] = dict(mean_epoch_seconds=float(np.mean([load(root / 'frozen-indexes' / (i + '-training-selection.json'))['optimizer_training_seconds'] / 30 for i in refs])))
    summary = dict(kernel_status='ERROR', whole_matrix_complete=False, requested_fits=48,
        completed_fits=len(rows), completed_architectures=complete, failures=suite['failures'],
        all_completed_fits_full_budget=True, all_amp_skips_zero=True, verified_prediction_files=len(checks),
        test_evaluated=False, calibration=calibration, sensitivity=sensitivity, timing=timing,
        worker_fit_wall_minutes={str(j): sum(r['selection']['training_seconds'] for r in w['results'].values()) / 60 for j, w in enumerate(workers)},
        scope='Exploratory eight-index calibration; two mini generator seeds, one coupled full-generator seed/index. Epoch choice is post-hoc. No independent transfer validation.')
    if (root / 'full-validation').is_dir():
        ids = c['index_ids']; full = []
        for i in ids:
            path = root / 'full-validation' / (i + '.npz')
            sid = np.load(root / 'frozen-indexes' / (i + '-semantic-ids.npy'), allow_pickle=False)
            with np.load(path, allow_pickle=False) as z:
                np.testing.assert_array_equal(z['user_ids'], cases[1]); np.testing.assert_array_equal(z['target_item_ids'], cases[2])
                hit = (z['semantic_ids'] == sid[cases[2], None, :]).all(-1)
                rank = np.where(hit.any(1), hit.argmax(1) + 1, 0)
                v = np.where((rank > 0) & (rank <= 10), 1 / np.log2(np.maximum(rank, 1) + 1), 0.)
            np.testing.assert_allclose(v.mean(), refs[i]['epochs']['30'], atol=1e-12, rtol=0)
            full.append(v)
        full = np.asarray(full); rng = np.random.default_rng(20261106)
        draws = []
        for _ in range(50): draws.append(rng.integers(0, 2786, size=(100, 2786)))
        uncertainty = {}
        for arch, e in [('narrow', '10'), ('narrow', '20'), ('shallow', '20')]:
            proxy = np.asarray([np.mean([scores[next(k for k, r in rows.items() if
                (r['architecture'], r['index_id'], r['seed']) == (arch, i, s)), e]
                for s in c['generator_seeds']], axis=0) for i in ids])
            correlations = []
            for sample in draws:
                f = full[:, sample].mean(-1).T; p = proxy[:, sample].mean(-1).T
                correlations.extend(rank_correlation(x, y) for x, y in zip(p, f))
            uncertainty[f'{arch}_{e}'] = dict(conditional_user_bootstrap_rho_ci95=np.quantile(correlations, [.025, .975]).tolist(), draws=5000)
        summary['conditional_user_uncertainty'] = dict(results=uncertainty,
            scope='Shared user resampling across fixed fitted models/indexes; excludes training/index selection uncertainty and target-item clustering. Not a proxy generalization CI.')
    log = load(root / 'rhpr-ml1m-mini-tiger.log')
    summary['kernel_last_log_seconds'] = max(r['time'] for r in log)
    for name, value in [('summary', summary), ('verification', checks), ('curves', curves)]:
        (out / (name + '.json')).write_text(json.dumps(value, indent=2) + '\n')
    with (out / 'fits.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    plot(summary, out)
    return summary


def plot(summary, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    colors = dict(shallow='#d95f02', narrow='#1b9e77', full='#666666')
    group = summary['calibration']['all']; comp = group['comparisons']
    for a in ['shallow', 'narrow', 'full']:
        vals = [comp[f'{a}_{e}']['mean']['spearman'] if a != 'full' else comp[f'{a}_{e}']['spearman'] for e in [5, 10, 15, 20]]
        axes[0].plot([5, 10, 15, 20], vals, marker='o', color=colors[a], label=a)
    axes[0].axhline(0, color='grey', linewidth=.6)
    axes[0].set(xlabel='Training epoch', ylabel='Spearman vs full epoch30 VAL NDCG@10', ylim=(-1, 1.05))
    axes[0].legend(); axes[0].grid(alpha=.2)
    row = comp['narrow_10']; target = group['target']; vals = row['mean_scores']
    for s, color in [(42, '#1b9e77'), (101, '#7570b3')]:
        ix = [j for j, i in enumerate(group['index_ids']) if i.endswith(f'seed{s}')]
        axes[1].scatter(np.asarray(target)[ix], np.asarray(vals)[ix], c=color, label=f'index seed{s}', s=60)
    for i, x, y in zip(group['index_ids'], target, vals): axes[1].annotate(i.replace('_seed', '\n'), (x, y), xytext=(4, 4), textcoords='offset points', fontsize=7)
    axes[1].set(xlabel='Full TIGER epoch30 VAL NDCG@10', ylabel='Narrow mini epoch10 VAL NDCG@10 (2 seeds)')
    axes[1].legend(); axes[1].grid(alpha=.2)
    fig.suptitle('Mini-TIGER pilot: 36/48 fits; shallow/narrow complete, compact partial')
    fig.tight_layout()
    for ext in ['png', 'pdf']: fig.savefig(out / ('calibration.' + ext), dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ['root', 'data', 'output']: p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args(); result = replay(a.root, a.data, a.output)
    print(json.dumps({k: result[k] for k in ['completed_fits', 'completed_architectures', 'verified_prediction_files', 'timing']}, indent=2))
