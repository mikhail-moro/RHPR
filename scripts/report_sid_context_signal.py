"""Report the CPU graph probes; no recommender validation/test evaluation."""
import argparse
import json
from pathlib import Path

import numpy as np

from rhpr.gr.data import load_beauty
from rhpr.gr.sid_context_signal import context_prototypes


def run(directory, plot=False):
    directory = Path(directory)
    config = json.loads((directory / 'config.json').read_text())
    prior = json.loads(Path(config['preparation_config']).read_text())
    curves = json.loads((directory / 'curves.json').read_text())
    assets = Path(config['als_input'])
    _, users, _, _ = load_beauty(Path(prior['data']))
    with np.load(assets / 'user-split.npz') as f:
        held_users = [users[i] for i in f['heldout_rows']]
    window = config['primary_window']
    primary = directory / f'window-{window}'
    factors = np.load(assets / 'als-item-factors.npy')
    mask = np.load(assets / 'cf-fit-user-frequency.npy') >= config['minimum_event_factor_users']
    held = context_prototypes(held_users, factors, window, mask)
    fit = np.load(primary / 'context-vectors.npy').astype(np.float64)
    with np.load(primary / 'support.npz') as f:
        support = f['users']
    a = fit / np.maximum(np.linalg.norm(fit, axis=1, keepdims=True), 1e-30)
    b = held.vectors.astype(np.float64)
    b /= np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-30)
    valid = (np.linalg.norm(fit, axis=1) > 0) & (held.contexts > 0) & (np.linalg.norm(b, axis=1) > 0)
    reliability = []
    for lo, hi in [(1, 1), (2, 4), (5, 9), (10, 19), (20, None)]:
        select = valid & (support >= lo) & (support <= hi if hi is not None else True)
        x, y = a[select], b[select]
        n = len(x)
        own_sum = float(np.einsum('ij,ij->', x, y))
        # Exact mean over all DIFFERENT item pairs within the FIT-support bin;
        # removes the trivial common-direction baseline without random sampling.
        baseline = float((x.sum(0) @ y.sum(0) - own_sum) / (n * (n - 1))) if n > 1 else None
        own = own_sum / n if n else None
        reliability.append(dict(fit_users_min=lo, fit_users_max=hi, comparable_items=n,
                                same_item_cosine=own, different_item_cosine=baseline,
                                identity_excess=own - baseline if baseline is not None else None))
    audit = dict(primary_window=window, reliability=reliability,
                 caveat='shared fixed ALS encoder; evidence for prototype stability, not downstream NIP quality')
    (directory / 'reliability-audit.json').write_text(json.dumps(audit, indent=2, allow_nan=False) + '\n')
    lines = ['# TRAIN-only context SID probe', '',
             'CPU only. Mean within preceding H, equal mean across target contexts. '
             'ALS factors frozen; disjoint TRAIN users for graph fitting and diagnostics. '
             'No TIGER fit, recommender validation or test evaluation.', '',
             '| Window | Index | Accepted swaps | Changed items | Heldout gain, bits | Context heldout LCP | Semantic LCP | Price R² |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    for w in config['windows']:
        rows = [r for r in curves if r['window'] == w]
        for variant in ['original', 'existing_item_als_uniform', 'context_als_uniform']:
            r = [r for r in rows if r['variant'] == variant][-1]
            lines.append(f"| {w} | {variant} | {r['accepted']} | {r['changed_items']} | "
                         f"{r['heldout_gain_bits']:.6f} | {r['context_heldout_score']:.6f} | "
                         f"{r['semantic_score']:.6f} | {r['price_r2']:.6f} |")
    lines += ['', 'These are graph/count-model diagnostics, not NDCG or a calibrated quality proxy. '
              'Confidence weighting also differs from the old item-ALS graph. '
              'Primary window was fixed at20; window5 is a sensitivity. '
              'No candidate is selected by recommender validation.', '',
              'Short H uses available events. No usable factors means the context is skipped. '
              'Fewer than2context users means no CF edges, not frozen SID ownership: '
              'these items can still donate paths through the semantic part of a move. '
              'Support reliability is based on distinct users, not epoch repetitions.', '',
              'The first weighted-cosine top-k probe is retained in the parent directory. '
              'Its graph favored popular neighbors; the current graph selects by cosine among '
              'eligible endpoints and applies reliability only to edge weights.', '',
              'All complete original paths, suffixes, occupancies and uniqueness are preserved. '
              'Means discard within-window order and can merge multiple contexts. '
              'Any generator comparison still needs matched fits; ALS20weights did not survive.']
    (directory / 'README.md').write_text('\n'.join(lines) + '\n')
    if plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
        fields = [('heldout_gain_bits', 'Heldout TRAIN gain (bits)'),
                  ('semantic_score', 'Semantic graph LCP'),
                  ('price_r2', 'Catalogue price R²'),
                  ('context_heldout_score', 'Heldout context LCP change (%)')]
        for ax, (field, label) in zip(axes.flat, fields):
            for w in config['windows']:
                subset = [r for r in curves if r['window'] == w]
                original = next(r for r in subset if r['variant'] == 'original')
                rows = [original] + [r for r in subset if r['variant'] == 'context_als_uniform']
                value = lambda r: 100 * (r[field] / original[field] - 1) if field == 'context_heldout_score' else r[field]
                ax.plot([r['accepted'] for r in rows], [value(r) for r in rows], marker='o', label=f'window {w}')
                control = next(r for r in subset if r['variant'] == 'existing_item_als_uniform')
                ax.scatter([control['accepted']], [value(control)], marker='x', color='black')
            ax.set(xlabel='Accepted context swaps', ylabel=label)
            ax.grid(alpha=.2)
        axes[0, 0].legend()
        fig.suptitle('CPU diagnostics only; black × = old item-ALS swaps')
        fig.savefig(directory / 'curves.png', dpi=140)
        plt.close(fig)
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', required=True)
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()
    run(args.directory, args.plot)
