"""Render fixed-epoch validation cohorts, intervals and descriptive trajectories."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--analysis', type=Path, default=Path('/workspace/artifacts/sid-context-screen/analysis'))
root = parser.parse_args().analysis
read = lambda n: json.loads((root / (n + '.json')).read_text())
cohorts, paired, stability = read('cohorts'), read('paired-epoch20'), read('epoch-stability')
models = ['rqkmeans_original', 'item_als', 'context_als_20', 'context_als_5']
names = ['Original', 'Item ALS', 'Context20', 'Context5']
labels = ['3', '4', '5-9', '10-19', '20+']
colors = ['black', '#377eb8', '#e68613', '#4daf4a']
value = lambda c, m, e=20: next(r for r in cohorts if r['cohort'] == c and r['model'] == m and r['epoch'] == e)
comparison = lambda c, m, b='rqkmeans_original': next(r for r in paired if r['cohort'] == c and r['comparison'] == m + '_vs_' + b)

lines = ['# Context-ALS screen: validation by TRAIN history length', '',
         'Four matched20epoch fits completed. Primary endpoint20;validation5/10/15/20. '
         'All16saved beam files verified against logged NDCG/Recall,exact users/targets, '
         'input indices,initialization and training order. No test or new GPU work.', '',
         '| TRAIN events | Users | Original | Item ALS | Context20 | Context5 |',
         '|---|---:|---:|---:|---:|---:|']
for label in ['all'] + labels:
    rows = [value(label, m) for m in models]
    lines.append('| ' + label + ' | ' + str(rows[0]['users']) + ' | ' + ' | '.join(f"{r['ndcg10']:.6f}" for r in rows) + ' |')
lines += ['', 'NDCG@10 at20epochs. TRAIN history includes all observed TRAIN events; '
          'the generator sees only min(length,20). Cohorts contain different users and '
          'targets,not truncated versions of the same inputs.', '',
          '| TRAIN events | Item ALS vs original | Context20 vs original | Context5 vs original |',
          '|---|---:|---:|---:|']
for label in ['all'] + labels:
    lines.append('| ' + label + ' | ' + ' | '.join(f"{comparison(label, m)['relative_percent']:+.2f}%" for m in models[1:]) + ' |')
lines += ['', '## Paired intervals at20epochs', '',
          '| Cohort | Comparison | Absolute delta | User95% interval | Target-cluster95% interval | Better/worse/equal users |',
          '|---|---|---:|---|---|---|']
for r in paired:
    fmt = lambda a: '[' + ', '.join(f'{v:+.6f}' for v in a) + ']'
    lines.append(f"| {r['cohort']} | {r['comparison']} | {r['delta']:+.6f} | {fmt(r['user_ci95'])} | "
                 f"{fmt(r['target_cluster_ci95'])} | {r['better']}/{r['worse']}/{r['equal']} |")
lines += ['', '2000paired resamples,seed2740+cohort index. User intervals resample the '
          'same users jointly across models. Target-cluster sensitivity resamples target '
          'items with all their validation users and uses the resulting variable-size '
          'mean. These are exploratory,unadjusted intervals conditional on one set '
          'of fitted weights;they do not measure variation across training seeds.', '',
          '## Long cohort through training', '',
          '| Comparison on20+ | Epoch5 | Epoch10 | Epoch15 | Epoch20 |',
          '|---|---:|---:|---:|---:|']
for r in stability:
    if r['cohort'] == '20+':
        lines.append('| ' + r['comparison'] + ' | ' + ' | '.join(f'{v:+.2f}%' for v in r['relative_percent']) + ' |')
lines += ['', 'The Context20endpoint improvement also appears under Item ALS. '
          'Its incremental advantage over Item ALS must be assessed separately; '
          'the larger gain versus original does not isolate the new context signal. '
          'Context5has a consistent positive sign on20+across these four checks, '
          'but checkpoints share users and fitted trajectories and are not replications.', '',
          'Mean-context coherence and the count-model gain improved in the CPU probe, '
          'while overall generator NDCG declined here. Those diagnostics are not '
          'validated monotonic proxies of retrieval quality. Semantic distortion and '
          'confidence policy also differ between swapped graphs. No routing threshold '
          'or new variant is chosen by these post-hoc cohort outcomes.', '',
          '![History cohorts at20epochs](history-length.png)', '',
          '![Training trajectories by cohort](epoch-history.png)', '',
          'Files:cohorts.json(all4epochs/all4ranking metrics),paired-epoch20.json, '
          'epoch-stability.json,user-metrics.parquet and user-metrics.npz(all16ranks '
          'plus history length,target support,path-change flags),training.json and '
          'verification.json. Rank999means absent from saved final20beams,not an exact '
          'full-catalogue rank. Raw outputs and observed hashes are in '
          '`/tmp/rhpr-context-screen-completed/`;large model states remain on private '
          'Kaggle. Re-run collect.py,analyze.py,report.py with PYTHONPATH pointing to '
          'the repository src;no fitting or inference is required.']
(root / 'README.md').write_text('\n'.join(lines) + '\n')
for name, rows in [('cohorts', cohorts), ('paired-epoch20', paired)]:
    with (root / (name + '.csv')).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
x = np.arange(len(labels))
for m, title, color in zip(models, names, colors):
    axes[0].plot(x, [value(c, m)['ndcg10'] for c in labels], marker='o', color=color, label=title)
for j, (m, title, color) in enumerate(zip(models[1:], names[1:], colors[1:])):
    rows = [comparison(c, m) for c in labels]
    means = np.array([r['delta'] for r in rows])
    intervals = np.array([r['user_ci95'] for r in rows])
    axes[1].errorbar(x + (j - 1) * .12, means, yerr=[means - intervals[:, 0], intervals[:, 1] - means],
                     fmt='o', capsize=3, color=color, label=title)
axes[0].set(ylabel='Validation NDCG@10', title='Epoch20: absolute quality')
axes[1].set(ylabel='NDCG@10 difference vs original', title='Paired user95% intervals (unadjusted)')
axes[1].axhline(0, color='black', linewidth=.8)
for ax in axes:
    ax.set_xticks(x, [f"{c}\n(n={value(c, models[0])['users']:,})" for c in labels])
    ax.set_xlabel('User TRAIN events; model sees at most20')
    ax.grid(alpha=.2)
axes[0].legend()
fig.savefig(root / 'history-length.png', dpi=145)
plt.close(fig)

fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
for ax, c in zip(axes.flat, ['all'] + labels):
    for m, title, color in zip(models, names, colors):
        ax.plot([5, 10, 15, 20], [value(c, m, e)['ndcg10'] for e in [5, 10, 15, 20]],
                marker='o', color=color, label=title)
    ax.set(title=f"{c} TRAIN events (n={value(c, models[0])['users']:,})", xlabel='Epoch', ylabel='NDCG@10')
    ax.set_xticks([5, 10, 15, 20]);ax.grid(alpha=.2)
axes[0, 0].legend(fontsize=8)
fig.savefig(root / 'epoch-history.png', dpi=145)
plt.close(fig)
print('\n'.join(lines[:25]))
