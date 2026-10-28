"""Best-of-two validation oracle for a frozen identical SID mapping."""
import json
from pathlib import Path

import numpy as np

from rhpr.gr.data import build_examples, file_sha256


def pair_scores(a, b):
    """Pick one complete ranking per example; do not merge or promote items."""
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or a.ndim != 1 or not len(a) or np.any(a < 1) or np.any(b < 1):
        raise ValueError('oracle requires nonempty matching positive rank vectors')
    def ndcg(r):
        return np.where(r <= 10, 1 / np.log2(r.astype(float) + 1), 0)
    x, y = ndcg(a), ndcg(b)
    oracle = np.maximum(x, y)
    best = max(float(x.mean()), float(y.mean()))
    return dict(a_ndcg10=float(x.mean()), b_ndcg10=float(y.mean()),
                oracle_ndcg10=float(oracle.mean()), best_fixed_ndcg10=best,
                absolute_gain=float(oracle.mean()-best),
                relative_gain_percent=float(100*(oracle.mean()/best-1)) if best else 0.0,
                relative_gain_defined=bool(best),
                a_hits=int((a <= 10).sum()), b_hits=int((b <= 10).sum()),
                a_only_hits=int(((a <= 10) & (b > 10)).sum()),
                b_only_hits=int(((a > 10) & (b <= 10)).sum()),
                both_hits=int(((a <= 10) & (b <= 10)).sum()),
                both_miss=int(((a > 10) & (b > 10)).sum()),
                oracle_recall10=float(np.mean((a <= 10) | (b <= 10))), users=len(a))


def saved_same_index_oracle(config, users, output):
    output = Path(output)
    trials = config['trials']
    if len(trials) != 2:
        raise ValueError('pair oracle requires exactly two fitted generators')
    ids = [t['id'] for t in trials]
    files = [output / 'indices' / name / 'semantic-ids.npy' for name in ids]
    if file_sha256(files[0]) != file_sha256(files[1]):
        raise ValueError('same-index oracle mappings are not byte-identical')
    table = np.load(files[0], allow_pickle=False)
    contexts, uids, targets = build_examples(users, config['data']['history_items'], 'validation')
    curves = [json.loads((output / 'trials' / name / 'training-curves.json').read_text()) for name in ids]
    epochs = [[r['epoch'] for r in c if 'validation' in r] for c in curves]
    expected = list(range(config['training']['validation_interval'], config['training']['max_epochs']+1,
                          config['training']['validation_interval']))
    if epochs[0] != expected or epochs[1] != expected:
        raise ValueError('oracle validation checkpoints differ from declared cadence')
    ranks, rows = dict(user_ids=uids, target_item_ids=targets), []
    for epoch in expected:
        pair = []
        for j, name in enumerate(ids):
            with np.load(output / 'trials' / name / f'validation-epoch-{epoch:03d}.npz') as f:
                np.testing.assert_array_equal(f['user_ids'], uids)
                np.testing.assert_array_equal(f['target_item_ids'], targets)
                predictions = f['semantic_ids']
                if predictions.shape != (len(uids), config['generator']['beam_width'], table.shape[1]):
                    raise ValueError('saved oracle beam dimensions differ')
                hit = np.all(predictions == table[targets, None, :], axis=-1)
                rank = np.where(hit.any(1), hit.argmax(1)+1, 999).astype(np.int16)
            score = pair_scores(rank, rank)['a_ndcg10']
            logged = next(r['validation']['ndcg@10'] for r in curves[j] if r['epoch'] == epoch)
            if abs(score-logged) > 1e-12:
                raise ValueError('independent oracle scoring differs from logged item metric')
            ranks[f'{name}_rank_e{epoch}'] = rank
            pair.append(rank)
        rows.append(dict(epoch=epoch, **pair_scores(*pair)))
    np.savez_compressed(output / 'oracle-user-ranks.npz', **ranks)
    report = dict(trials=ids, generator_seeds=[t['generator_seed'] for t in trials],
                  data_order_seed=config['seed'], sid_file_sha256=file_sha256(files[0]),
                  definition='Per-example max NDCG10 of two complete lists; target-aware upper bound only.',
                  epochs=rows, test_evaluations=0, new_inference=False,
                  caveat='One fitted pair, shared users across epochs; no learned-router or causal index-gain claim.')
    (output / 'paired-oracle.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report
