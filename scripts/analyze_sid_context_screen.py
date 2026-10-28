"""Paired history-length analysis of saved validation beams; CPU-only."""
import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from rhpr.gr.data import build_examples, file_sha256, load_beauty
from rhpr.gr.generator_suite import training_manifest

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--completed', type=Path, default=Path('/tmp/rhpr-context-screen-completed'))
parser.add_argument('--output', type=Path, default=Path('/workspace/artifacts/sid-context-screen/analysis'))
parser.add_argument('--data', type=Path, default=Path('/workspace/scratch/gr-beauty-input/data'))
parser.add_argument('--assets', type=Path, default=Path('/workspace/artifacts/sid-context-screen-input'))
parser.add_argument('--repo', type=Path, default=Path('/workspace/RHPR'))
args = parser.parse_args()
COMPLETED, OUTPUT, REPO = args.completed, args.output, args.repo
OUTPUT.mkdir(parents=True, exist_ok=True)
MODELS = ['rqkmeans_original', 'item_als', 'context_als_20', 'context_als_5']
EPOCHS = [5, 10, 15, 20]
REPEATS, SEED = 2000, 2740
load = lambda p: json.loads(p.read_text())
write = lambda name, data: (OUTPUT / (name + '.json')).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')

manifest = load(COMPLETED / 'run-manifest.json')
config = manifest['config']
assert manifest['git_sha'] == '9dcee38f1fe77cff18daf093246c7c81e9dc0751'
assert config == load(REPO / 'experiments/amazon_beauty_sid_context_screen_2t4.json')
suite = load(COMPLETED / 'suite-results.json')
assert not suite['failures'] and set(suite['results']) == set(MODELS)
_, users, ids, _ = load_beauty(args.data, config['data'])
contexts, uids, targets = build_examples(users, 20, 'validation')
lengths = np.array([len(u.train) for u in users])
visible = (contexts >= 0).sum(1)
np.testing.assert_array_equal(visible, np.minimum(lengths, 20))
assert load(COMPLETED / 'shared-training-stream.json') == training_manifest(users, config)
assert load(COMPLETED / 'initialization.json')['initial_state_sha256'] == config['expected_initial_state_sha256']
asset_root = args.assets
assets = load(asset_root / 'resources-manifest.json')
assert file_sha256(asset_root / 'resources-manifest.json') == config['frozen_indices']['manifest_sha256']
paths, rank, training = {}, {}, []
for trial in config['trials']:
    name = trial['id']
    origin = assets['indices'][trial['asset_id']]
    filename = origin['original_sids'] if trial.get('use_original_paths') else origin['sids']
    path = COMPLETED / 'indices' / name / 'semantic-ids.npy'
    assert file_sha256(path) == file_sha256(asset_root / filename)
    paths[name] = np.load(path)
    result = suite['results'][name]
    selection = result['selection']
    assert result['initial_state_sha256'] == config['expected_initial_state_sha256']
    assert result['test_evaluations'] == 0 and not result['test']
    assert not result['quantizer_retrained']
    assert selection['epochs'] == 20 and selection['validation_checks'] == 4
    assert selection['model_examples_consumed'] == 2628260
    assert selection['optimizer_updates'] + selection['amp_skipped_updates'] == 10280
    assert selection['training_precision'] == 'fp16' and selection['evaluation_precision'] == 'fp32'
    training.append(dict(model=name, epochs=20, updates=selection['optimizer_updates'],
                         amp_skips=selection['amp_skipped_updates'],
                         train_seconds=selection['training_seconds'], validation_seconds=selection['validation_seconds']))
    curve = load(COMPLETED / 'trials' / name / 'training-curves.json')
    assert [r['epoch'] for r in curve if 'validation' in r] == EPOCHS
    for epoch in EPOCHS:
        with np.load(COMPLETED / 'trials' / name / f'validation-epoch-{epoch:03d}.npz') as f:
            np.testing.assert_array_equal(f['user_ids'], uids)
            np.testing.assert_array_equal(f['target_item_ids'], targets)
            assert f['semantic_ids'].shape == (len(uids), 20, 4) and np.isfinite(f['scores']).all()
            hit = np.all(f['semantic_ids'] == paths[name][targets, None, :], axis=-1)
            ranks = np.where(hit.any(1), hit.argmax(1) + 1, 999).astype(np.int16)
        rank[name, epoch] = ranks
        declared = curve[epoch - 1]['validation']
        for cutoff in [5, 10]:
            observed = np.where(ranks <= cutoff, 1 / np.log2(ranks.astype(np.float64) + 1), 0)
            assert abs(observed.mean() - declared[f'ndcg@{cutoff}']) < 1e-12
            assert abs((ranks <= cutoff).mean() - declared[f'recall@{cutoff}']) < 1e-12

metrics = {k: np.where(v <= 10, 1 / np.log2(v.astype(np.float64) + 1), 0) for k, v in rank.items()}
cohorts = [('all', lengths >= 0), ('3', lengths == 3), ('4', lengths == 4),
           ('5-9', (lengths >= 5) & (lengths <= 9)),
           ('10-19', (lengths >= 10) & (lengths <= 19)), ('20+', lengths >= 20)]
assert sum(int(mask.sum()) for _, mask in cohorts[1:]) == len(uids)
comparisons = [('item_als', 'rqkmeans_original'), ('context_als_20', 'rqkmeans_original'),
               ('context_als_5', 'rqkmeans_original'), ('context_als_20', 'item_als'),
               ('context_als_5', 'item_als')]


def intervals(matrix, labels, seed):
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(REPEATS // 40):
        ix = rng.integers(len(matrix), size=(40, len(matrix)))
        draws.append(matrix[ix].mean(1))
    user = np.quantile(np.concatenate(draws), [.025, .975], axis=0)
    _, group = np.unique(labels, return_inverse=True)
    groups = group.max() + 1
    count = np.bincount(group)
    summed = np.stack([np.bincount(group, weights=matrix[:, i]) for i in range(matrix.shape[1])], axis=1)
    draws = []
    for _ in range(REPEATS // 40):
        ix = rng.integers(groups, size=(40, groups))
        draws.append(summed[ix].sum(1) / count[ix].sum(1)[:, None])
    item = np.quantile(np.concatenate(draws), [.025, .975], axis=0)
    return user, item


rows, paired, stability = [], [], []
for c, (label, mask) in enumerate(cohorts):
    for epoch in EPOCHS:
        for model in MODELS:
            r = rank[model, epoch][mask]
            rows.append(dict(cohort=label, users=int(mask.sum()), epoch=epoch, model=model,
                             mean_train_history=float(lengths[mask].mean()),
                             mean_visible_context=float(visible[mask].mean()),
                             ndcg10=float(metrics[model, epoch][mask].mean()),
                             recall10=float((r <= 10).mean()), hits10=int((r <= 10).sum()),
                             ndcg5=float(np.where(r <= 5, 1 / np.log2(r.astype(np.float64) + 1), 0).mean()),
                             recall5=float((r <= 5).mean())))
    delta = np.stack([metrics[a, 20][mask] - metrics[b, 20][mask] for a, b in comparisons], axis=1)
    user_ci, item_ci = intervals(delta, targets[mask], SEED + c)
    for j, (a, b) in enumerate(comparisons):
        baseline = metrics[b, 20][mask].mean()
        paired.append(dict(cohort=label, users=int(mask.sum()), comparison=a + '_vs_' + b,
                           delta=float(delta[:, j].mean()), relative_percent=float(100 * delta[:, j].mean() / baseline),
                           user_ci95=user_ci[:, j].tolist(), target_cluster_ci95=item_ci[:, j].tolist(),
                           summed_delta=float(delta[:, j].sum()),
                           all_user_ndcg_contribution=float(delta[:, j].sum() / len(uids)),
                           better=int((delta[:, j] > 0).sum()), worse=int((delta[:, j] < 0).sum()),
                           equal=int((delta[:, j] == 0).sum())))
        differences = np.stack([metrics[a, e][mask] - metrics[b, e][mask] for e in EPOCHS], axis=1)
        means = differences.mean(0)
        stability.append(dict(cohort=label, comparison=a + '_vs_' + b,
                              epochs=EPOCHS, absolute_delta=means.tolist(),
                              positive_epochs=int((means > 0).sum()),
                              relative_percent=[float(100 * v / metrics[b, e][mask].mean()) for e, v in zip(EPOCHS, means)],
                              late10_15_20_mean_delta=float(means[1:].mean()),
                              late10_15_20_user_positive_atleast2=int(((differences[:, 1:] > 0).sum(1) >= 2).sum()),
                              late10_15_20_user_negative_atleast2=int(((differences[:, 1:] < 0).sum(1) >= 2).sum())))

train_positions = np.zeros(len(ids), dtype=np.int64)
train_users = train_positions.copy()
for user in users:
    np.add.at(train_positions, np.asarray(user.train[1:]), 1)
    train_users[np.unique(user.train)] += 1
data = dict(user_id=uids, target_item_id=targets, train_history_length=lengths,
            visible_context_length=visible, target_train_positions=train_positions[targets],
            target_train_users=train_users[targets])
for model in MODELS:
    changed = np.any(paths[model] != paths['rqkmeans_original'], axis=1)
    data[model + '_target_path_changed'] = changed[targets]
    data[model + '_context_path_changed'] = np.any(changed[np.maximum(contexts, 0)] & (contexts >= 0), axis=1)
    for epoch in EPOCHS:
        data[model + f'_rank_e{epoch}'] = rank[model, epoch]
pq.write_table(pa.table(data), OUTPUT / 'user-metrics.parquet', compression='zstd')
np.savez_compressed(OUTPUT / 'user-metrics.npz', **data)
write('cohorts', rows)
write('paired-epoch20', paired)
write('epoch-stability', stability)
write('training', training)
write('verification', dict(source_commit=manifest['git_sha'], config_verified=True,
                           training_stream_and_neural_initialization_verified=True,
                           exact_original_and_swapped_index_bytes_verified=True,
                           all16_validation_files_user_target_order_and_metrics_verified=True,
                           users=len(uids), epochs=EPOCHS, trials=MODELS, test_evaluations=0,
                           bootstrap_repeats=REPEATS, bootstrap_seed=SEED,
                           history_min=int(lengths.min()), history_max=int(lengths.max()),
                           caveat='Post-hoc cohorts,unadjusted conditional bootstrap intervals;one fitted seed,not training uncertainty.' ))
for r in paired:
    if r['comparison'].endswith('_vs_rqkmeans_original'):
        print(json.dumps(r))
