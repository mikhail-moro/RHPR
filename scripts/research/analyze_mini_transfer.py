"""Independent CPU replay of the fixed narrow/epoch10 seed202 transfer check."""
import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from rhpr.gr.mini_tiger import ARCHITECTURES, validate
from rhpr.gr.movie_data import case_digest
from rhpr.gr.sid import SidIndex


def load(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def decode(path, sids, cases, expected):
    with np.load(path, allow_pickle=False) as z:
        np.testing.assert_array_equal(z['user_ids'], cases[1])
        np.testing.assert_array_equal(z['target_item_ids'], cases[2])
        pred = z['semantic_ids']; assert pred.shape == (2786, 20, 4)
        assert np.isfinite(z['scores']).all() and np.all((pred >= 0) & (pred < 256))
        hit = (pred == sids[cases[2], None, :]).all(-1)
        rank = np.where(hit.any(1), hit.argmax(1) + 1, 0)
        weights = np.asarray([256**3, 256**2, 256, 1])
        catalogue = np.sort(sids @ weights); keys = pred @ weights
        pos = np.searchsorted(catalogue, keys)
        valid = (pos < len(catalogue)) & (catalogue[np.minimum(pos, len(catalogue) - 1)] == keys)
    metrics = {}
    for k in [5, 10]:
        ok = (rank > 0) & (rank <= k)
        metrics[f'ndcg@{k}'] = np.where(ok, 1 / np.log2(np.maximum(rank, 1) + 1), 0.)
        metrics[f'recall@{k}'] = ok.astype(float)
    for k, v in metrics.items(): np.testing.assert_allclose(v.mean(), expected[k], atol=1e-12, rtol=0)
    invalid = int((~valid).sum()); dup = sum(len(k[v]) - len(np.unique(k[v])) for k, v in zip(keys, valid))
    assert invalid == expected['invalid_predictions'] and dup == expected['duplicate_predictions']
    return metrics['ndcg@10'], dict(sha256=sha(path), invalid=invalid, duplicates=int(dup))


def replay(root, data, out):
    out.mkdir(parents=True, exist_ok=True)
    c = load(root / 'experiment.json'); validate(c)
    assert c['calibration_phase'] == 'seed202_transfer'
    manifest = load(root / 'run-manifest.json'); plan = load(root / 'plan.json')
    assert manifest['config'] == plan['config'] == c
    assert manifest['git_sha'] == '31d120c8c962adfb7c9f1f4db30e18c1f8d6d153'
    metrics = load(root / 'metrics.json')
    assert metrics['completed'] == 8 and metrics['indexes'] == 4 and metrics['architectures'] == 1
    assert metrics['test_evaluated'] is False
    assert all(np.isfinite(v) for v in metrics.values() if isinstance(v, (int, float)))
    assert (root / 'summary.md').is_file()
    suite = load(root / 'suite-results.json'); assert not suite['failures'] and len(suite['results']) == 8
    workers = [load(p) for p in sorted((root / 'workers').glob('worker-*.json'))]
    assert len(workers) == 2 and all(len(w['results']) == 4 and not w['failures'] for w in workers)
    assert {k: v for w in workers for k, v in w['results'].items()} == suite['results']
    for split in ['train', 'validation']:
        with np.load(data / (split + '-cases.npz'), allow_pickle=False) as z:
            values = tuple(z[k] for k in ['contexts', 'user_ids', 'targets'])
        assert case_digest(values) == c['data']['case_sha256'][split]
        if split == 'validation': cases = values
    refs = load(root / 'frozen-indexes/references.json'); checks = []; records = []; users = {}
    full_users = []; target = []; full_curve = {}; rows = suite['results']
    for i in c['index_ids']:
        assert refs[i]['full_seed'] == 202
        for f, digest in c['parents']['202']['artifacts'][i].items():
            assert sha(root / 'frozen-indexes' / (i + '-' + f)) == digest
        sid = np.load(root / 'frozen-indexes' / (i + '-semantic-ids.npy'), allow_pickle=False)
        assert sid.shape == (2874, 4); SidIndex.from_full_sids(sid)
        fc = load(root / 'frozen-indexes' / (i + '-training-curves.json'))
        full_curve[i] = {str(r['epoch']): r['validation'] for r in fc if 'validation' in r}
        assert refs[i]['epochs'] == {e: v['ndcg@10'] for e, v in full_curve[i].items()}
        v, ch = decode(root / 'full-validation' / (i + '.npz'), sid, cases, full_curve[i]['30'])
        full_users.append(v); target.append(v.mean()); checks.append(dict(path='full-validation/' + i + '.npz', **ch))
        for s in c['generator_seeds']:
            name = f'narrow_{i}_gen{s}'; p = root / 'trials' / name; r = load(p / 'trial-result.json')
            assert r == rows[name] and (r['architecture'], r['index_id'], r['seed']) == ('narrow', i, s)
            g = load(p / 'generator.json'); sel = load(p / 'training-selection.json')
            assert g['initial_state_sha256'] == c['initialization_pins']['narrow'][str(s)]
            assert g['config'] == {**c['generator'], **ARCHITECTURES['narrow']}
            assert g['sids_sha256'] == refs[i]['sids_sha256'] and g['parameters'] == 1375040
            assert sel['training_epochs'] == 10 and sel['optimizer_updates'] == 2960
            assert sel['validation_checks'] == 2 and sel['amp_skipped_updates'] == 0
            assert sel['training_examples_per_epoch'] == 151166
            assert sel['training_precision'] == 'fp16' and sel['evaluation_precision'] == 'fp32'
            curve = load(p / 'training-curves.json'); assert len(curve) == 10
            assert all(x['updates'] == x['epoch'] * 296 and x['train_examples'] == 151166 for x in curve)
            assert [x['epoch'] for x in curve if 'validation' in x] == [5, 10]
            for x in curve:
                if 'validation' not in x: continue
                path = p / f"validation-epoch-{x['epoch']:03d}.npz"
                v, ch = decode(path, sid, cases, x['validation']); checks.append(dict(path=str(path.relative_to(root)), **ch))
                assert r['epochs'][str(x['epoch'])] == x['validation']['ndcg@10']
                users[i, s, x['epoch']] = v
            records.append(dict(index_id=i, generator_seed=s, epoch10_ndcg=float(users[i, s, 10].mean()),
                mean_epoch_seconds=sel['optimizer_training_seconds'] / 10, fit_seconds=sel['training_seconds']))
    full_users = np.asarray(full_users)
    proxy_users = np.asarray([np.mean([users[i, s, 10] for s in c['generator_seeds']], axis=0) for i in c['index_ids']])
    proxy = proxy_users.mean(1); rho = float(spearmanr(proxy, target).statistic)
    saved = load(root / 'correlations.json'); primary = saved['primary']
    np.testing.assert_allclose(rho, primary['mean']['spearman'], atol=1e-12, rtol=0)
    np.testing.assert_allclose(rho, metrics['narrow_epoch10_spearman'], atol=1e-12, rtol=0)
    np.testing.assert_allclose(proxy, primary['mean_scores'], atol=1e-12, rtol=0)
    per_seed = {}
    for s in c['generator_seeds']:
        scores = [users[i, s, 10].mean() for i in c['index_ids']]
        per_seed[str(s)] = dict(spearman=float(spearmanr(scores, target).statistic), scores=scores)
        np.testing.assert_allclose(per_seed[str(s)]['spearman'], primary['seeds'][str(s)]['spearman'], atol=1e-12)
    rng = np.random.default_rng(20261106); boot = []
    for _ in range(50):
        sample = rng.integers(0, 2786, size=(100, 2786))
        for p, f in zip(proxy_users[:, sample].mean(-1).T, full_users[:, sample].mean(-1).T):
            boot.append(float(spearmanr(p, f).statistic))
    null = [float(spearmanr(proxy, v).statistic) for v in itertools.permutations(target)]
    table = [dict(index_id=i, full_epoch30_ndcg=float(target[j]), mini_mean_epoch10_ndcg=float(proxy[j]),
                  mini31415_epoch10_ndcg=float(users[i, 31415, 10].mean()), mini27182_epoch10_ndcg=float(users[i, 27182, 10].mean()))
             for j, i in enumerate(c['index_ids'])]
    result = dict(complete=True, fits=8, test_evaluated=False, source_code_sha=manifest['git_sha'],
        finished_at=manifest['finished_at'], run_minutes=manifest['duration_seconds'] / 60,
        verified_mini_prediction_files=16, verified_full_prediction_files=4, amp_skips=0,
        spearman=rho, per_seed=per_seed, primary=saved['primary'], table=table,
        early_full=saved['groups']['all']['early_full'],
        conditional_user_bootstrap_ci95=np.quantile(boot, [.025, .975]).tolist(),
        bootstrap_scope='5000 shared user draws;fixed fitted models/four related indexes;excludes training randomness,index selection and target-item dependence',
        exact_label_permutation_two_sided_p=sum(abs(v) >= abs(rho) - 1e-12 for v in null) / 24,
        mean_epoch_seconds=float(np.mean([r['mean_epoch_seconds'] for r in records])),
        mean_fit_seconds=float(np.mean([r['fit_seconds'] for r in records])),
        scope='Architecture/epoch fixed before this proxy transfer run. Four related indexes,one full generator seed/index. Existing same VALcohort,not a universal or independently replicated proxy.')
    for name, value in [('summary', result), ('verification', checks)]:
        (out / (name + '.json')).write_text(json.dumps(value, indent=2) + '\n')
    with (out / 'fits.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(records[0])); w.writeheader(); w.writerows(records)
    plot(result, out)
    return result


def plot(result, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5)); rows = result['table']
    for key, label, color in [('mini31415_epoch10_ndcg', 'mini seed31415', '#7570b3'),
                               ('mini27182_epoch10_ndcg', 'mini seed27182', '#d95f02'),
                               ('mini_mean_epoch10_ndcg', 'two-seed average', '#1b9e77')]:
        axes[0].scatter([r['full_epoch30_ndcg'] for r in rows], [r[key] for r in rows], label=label, c=color, s=55)
    for r in rows: axes[0].annotate(r['index_id'].replace('_seed202', ''), (r['full_epoch30_ndcg'], r['mini_mean_epoch10_ndcg']), xytext=(4, 4), textcoords='offset points', fontsize=8)
    axes[0].set(xlabel='Full TIGER epoch30 VAL NDCG@10', ylabel='Fixed narrow mini epoch10 VAL NDCG@10')
    axes[0].legend(); axes[0].grid(alpha=.2)
    axes[1].bar(['seed31415', 'seed27182', 'mean'], [result['per_seed']['31415']['spearman'], result['per_seed']['27182']['spearman'], result['spearman']], color=['#7570b3', '#d95f02', '#1b9e77'])
    axes[1].axhline(0, c='grey', lw=.6); axes[1].set(ylabel='Spearman across four seed202 indexes', ylim=(-1, 1.1))
    fig.suptitle('Fixed epoch10 transfer: mean preserves top2; single-seed ranks disagree')
    fig.tight_layout()
    for ext in ['png', 'pdf']: fig.savefig(out / ('transfer.' + ext), dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ['root', 'data', 'output']: p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args(); r = replay(a.root, a.data, a.output)
    print(json.dumps({k: r[k] for k in ['fits', 'run_minutes', 'spearman', 'conditional_user_bootstrap_ci95', 'exact_label_permutation_two_sided_p', 'mean_epoch_seconds']}, indent=2))
