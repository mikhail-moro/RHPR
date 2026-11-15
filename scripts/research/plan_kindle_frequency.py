"""TRAIN-only frequency policy; measure merge loss and unpadded/padded FLOPs."""
import argparse
import json
from pathlib import Path

import numpy as np

from rhpr.gr.data import file_sha256
from rhpr.gr.kindle_frequency import compact20, matching_tables, train_support


def main():
    p = argparse.ArgumentParser(); p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); a = p.parse_args()
    support = train_support(a.data, 16177)
    costs = json.loads(Path('docs/gr-kindle-mixed20-preflight.json').read_text())['native_costs']
    # Supported matmul FLOPs are quadratic in encoder length for fixed decoder.
    counts = np.arange(6); values = np.array([costs[str(k)]['supported_forward_flops_batch1'] for k in counts])
    poly = np.polyfit(counts, values, 2)
    assert np.max(np.abs(np.polyval(poly, counts)-values)) < .01
    rows = []; budget = {}
    for split in ('train', 'validation', 'test'):
        z = np.load(a.data/f'{split}-cases.npz'); h, lengths = compact20(z['contexts'])
        rare = support[np.maximum(h, 0)] <= 5
        for fraction in (.5, .6, .7, .8, .9, 1.):
            old = np.floor(lengths*fraction+1e-8).astype(int)
            maximum = np.zeros(len(h), dtype=int); blocked = np.zeros(len(h), dtype=bool)
            for j in range(19):
                take = (j+1 < old) & ~rare[:, j] & ~rare[:, j+1] & ~blocked
                maximum += take; blocked = take
            strict_cost = np.polyval(poly, maximum)
            batch_cost = sum(len(part)*np.polyval(poly, int(part.min()))
                for part in np.array_split(maximum, np.arange(64, len(h), 64)))/len(h)
            rows.append(dict(split=split, eligible_fraction=fraction, cases=len(h),
                strict_mean_pairs=float(maximum.mean()), strict_zero_pair_fraction=float(np.mean(maximum == 0)),
                strict_five_pair_feasible_fraction=float(np.mean(maximum >= 5)),
                mean_encoder_positions_including_user=float((81-4*maximum).mean()),
                mean_unpadded_forward_gflops=float(strict_cost.mean()/1e9),
                native_order_batch64_padded_forward_gflops=float(batch_cost/1e9)))
        edge_cost = rare[:, :15].astype(np.int16)+rare[:, 1:16]
        cost, ways = matching_tables(edge_cost, lengths*4//5)
        k = np.minimum(5, (lengths*4//5)//2)
        minimum = cost[np.arange(len(h)), 0, k]
        budget[split] = dict(cases=len(h), mean_pairs=float(k.mean()),
            mean_rare_items_forced_inside_pairs=float(minimum.mean()),
            all_rare_context_items_preserved_fraction=float(np.mean(minimum == 0)),
            mean_native_full_window_forward_gflops=float(np.polyval(poly, k).mean()/1e9),
            pair_counts=np.bincount(k, minlength=6).tolist(),
            forced_rare_histogram=np.bincount(minimum, minlength=11).tolist())
    result = dict(uses_future_counts=False, threshold_selected_without_quality_metrics=5,
        frequency_definition='unique users in original TRAIN events, including first item, before prefix expansion',
        train_support_sha256=__import__('hashlib').sha256(support.astype('<i4').tobytes()).hexdigest(),
        data_sha256={f: file_sha256(a.data/f) for f in ('histories.npz', 'train-cases.npz', 'validation-cases.npz', 'test-cases.npz')},
        catalog_items=len(support), rare_items=int((support <= 5).sum()),
        rare_catalog_fraction=float(np.mean(support <= 5)),
        native_flop_quadratic_in_pair_count=poly.tolist(),
        flop_scope='saved B1 CPU math-SDPA supported forward, teacher-forced4codes, FMA2; no beam/backward/ST5',
        strict_rows=rows, fixed_budget_80percent=budget,
        note='Unpadded per-query mean is not batched latency; padding costs reported separately. No NDCG-based policy selection.')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
