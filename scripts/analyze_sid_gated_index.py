"""Independently decode saved gated SID beams and analyze paired validation users."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from rhpr.gr.data import build_examples, file_sha256, load_beauty
from rhpr.gr.generator_suite import training_manifest

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--completed', type=Path, default=Path('/tmp/rhpr-gated-index-completed'))
parser.add_argument('--output', type=Path, default=Path('/workspace/artifacts/sid-gated-index/analysis'))
parser.add_argument('--data', type=Path, default=Path('/workspace/scratch/gr-beauty-input/data'))
parser.add_argument('--assets', type=Path, default=Path('/workspace/artifacts/sid-context-screen-input'))
parser.add_argument('--plot', action='store_true')
args = parser.parse_args()
root, out = args.completed, args.output
out.mkdir(parents=True, exist_ok=True)
repo = Path(__file__).resolve().parents[1]
load = lambda p: json.loads(p.read_text())
write = lambda name, data: (out / (name + '.json')).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
manifest = load(root / 'run-manifest.json')
c = manifest['config']
assert manifest['git_sha'] == '45d7dab1d35a55821585e17423f32eb197a5cce3'
assert c == load(repo / 'experiments/amazon_beauty_sid_gated_index_2t4.json')
models = ['original_with_branch', 'gated_original_als']
epochs = [5, 10, 15, 20]
suite = load(root / 'suite-results.json')
assert not suite['failures'] and set(suite['results']) == set(models)
_, users, source_ids, counts = load_beauty(args.data, c['data'])
contexts, uids, targets = build_examples(users, 20, 'validation')
lengths = np.array([len(u.train) for u in users])
visible = (contexts >= 0).sum(1)
branch = (visible >= c['index_gate']['threshold']).astype(np.int64)
np.testing.assert_array_equal(visible, np.minimum(lengths, 20))
stream = training_manifest(users, c)
assert load(root / 'shared-training-stream.json') == stream
assert stream['train_cases_sha256'] == c['expected_train_cases_sha256']
assert stream['index_gate']['short_examples'] == 98608 and stream['index_gate']['long_examples'] == 32805
assert load(root / 'initialization.json')['initial_state_sha256'] == c['expected_initial_state_sha256']
assets = load(args.assets / 'resources-manifest.json')
assert file_sha256(args.assets / 'resources-manifest.json') == c['frozen_indices']['manifest_sha256']
original = np.load(args.assets / 'original.npy')
width = c['quantizer']['codebook_size']
pack_weights = width ** np.arange(3, -1, -1, dtype=np.int64)
valid_paths = original @ pack_weights
ranks, values, tables, training, verified = {}, {}, {}, [], []
for trial in c['trials']:
    name = trial['id']
    asset = assets['indices'][trial['asset_id']]
    expected = asset['original_sids'] if trial.get('use_original_paths') else asset['sids']
    path = root / 'indices' / name / 'semantic-ids.npy'
    assert file_sha256(path) == file_sha256(args.assets / expected)
    tables[name] = np.stack([original, np.load(path)])
    provenance = load(root / 'trials' / name / 'conditional-index.json')
    assert provenance['threshold'] == 10
    assert provenance['sid_matrix_sha256'] == hashlib.sha256(tables[name].tobytes()).hexdigest()
    result = suite['results'][name]
    selection = result['selection']
    assert result['initial_state_sha256'] == c['expected_initial_state_sha256']
    assert result['parameters'] == 4978176 and result['test_evaluations'] == 0 and not result['test']
    assert selection['epochs'] == 20 and selection['validation_checks'] == 4
    assert selection['model_examples_consumed'] == 2628260
    assert selection['optimizer_updates'] + selection['amp_skipped_updates'] == 10280
    training.append(dict(model=name, epochs=20, parameters=result['parameters'],
                         updates=selection['optimizer_updates'], amp_skips=selection['amp_skipped_updates'],
                         training_seconds=selection['training_seconds'], validation_seconds=selection['validation_seconds']))
    curve = load(root / 'trials' / name / 'training-curves.json')
    assert [r['epoch'] for r in curve if 'validation' in r] == epochs
    for epoch in epochs:
        file = root / 'trials' / name / f'validation-epoch-{epoch:03d}.npz'
        with np.load(file) as f:
            np.testing.assert_array_equal(f['user_ids'], uids)
            np.testing.assert_array_equal(f['target_item_ids'], targets)
            np.testing.assert_array_equal(f['branch_ids'], branch)
            np.testing.assert_array_equal(f['observed_context_lengths'], visible)
            predictions = f['semantic_ids']
            assert predictions.shape == (len(uids), 20, 4) and np.isfinite(f['scores']).all()
            assert np.all((predictions >= 0) & (predictions < width))
            # Decode each row with the independently recomputed available-prefix
            # branch, not the whole model's saved LONGtable or cached metric.
            truth = tables[name][branch, targets]
            hit = np.all(predictions == truth[:, None, :], axis=-1)
            rank = np.where(hit.any(1), hit.argmax(1) + 1, 999).astype(np.int16)
            packed = predictions @ pack_weights
            valid = np.isin(packed, valid_paths)
            ordered = np.sort(np.where(valid, packed, -1), axis=1)
            duplicate = (ordered[:, 1:] == ordered[:, :-1]) & (ordered[:, 1:] >= 0)
        ranks[name, epoch] = rank
        score = curve[epoch - 1]['validation']
        assert int((~valid).sum()) == score['invalid_predictions']
        assert int(duplicate.sum()) == score['duplicate_predictions']
        for cutoff in [5, 10]:
            for metric in ['ndcg', 'recall']:
                x = np.where(rank <= cutoff, 1 / np.log2(rank.astype(np.float64) + 1), 0) if metric == 'ndcg' else (rank <= cutoff).astype(float)
                values[name, epoch, f'{metric}{cutoff}'] = x
                assert abs(float(x.mean()) - score[f'{metric}@{cutoff}']) < 1e-12
        for b, label in [(0, 'short'), (1, 'long')]:
            mask = branch == b
            assert int(mask.sum()) == score[f'gate_{label}_examples']
            assert abs(float(values[name, epoch, 'ndcg10'][mask].mean()) - score[f'gate_{label}_ndcg10']) < 1e-12
        verified.append(dict(path=file.relative_to(root).as_posix(), sha256=file_sha256(file), branch_aware_metrics_match=True))

cohorts = [('all', lengths >= 0), ('<10', branch == 0), ('10+', branch == 1),
           ('3', lengths == 3), ('4', lengths == 4), ('5-9', (lengths >= 5) & (lengths <= 9)),
           ('10-19', (lengths >= 10) & (lengths <= 19)), ('20+', lengths >= 20)]


def intervals(delta, labels, seed):
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(50):
        ix = rng.integers(len(delta), size=(40, len(delta)))
        boot.extend(delta[ix].mean(1))
    user_ci = np.quantile(boot, [.025, .975]).tolist()
    _, groups = np.unique(labels, return_inverse=True)
    n = np.bincount(groups)
    summed = np.bincount(groups, weights=delta)
    boot = []
    for _ in range(50):
        ix = rng.integers(len(n), size=(40, len(n)))
        boot.extend(summed[ix].sum(1) / n[ix].sum(1))
    return user_ci, np.quantile(boot, [.025, .975]).tolist()


cohort_rows, paired, stability = [], [], []
for i, (label, mask) in enumerate(cohorts):
    for epoch in epochs:
        for model in models:
            row = dict(cohort=label, users=int(mask.sum()), epoch=epoch, model=model)
            row.update({metric: float(values[model, epoch, metric][mask].mean())
                        for metric in ['ndcg10', 'recall10', 'ndcg5', 'recall5']})
            cohort_rows.append(row)
    a, b = [values[model, 20, 'ndcg10'][mask] for model in models]
    delta = b - a
    ci, cluster_ci = intervals(delta, targets[mask], 2748 + i)
    paired.append(dict(cohort=label, users=int(mask.sum()), original=float(a.mean()), gated=float(b.mean()),
                       delta=float(delta.mean()), relative_percent=float(100 * delta.mean() / a.mean()),
                       user_ci95=ci, target_cluster_ci95=cluster_ci,
                       better=int((delta > 0).sum()), worse=int((delta < 0).sum()), equal=int((delta == 0).sum()),
                       all_user_delta_contribution=float(delta.sum() / len(uids))))
    diffs = np.stack([values[models[1], e, 'ndcg10'][mask] - values[models[0], e, 'ndcg10'][mask] for e in epochs], axis=1)
    stability.append(dict(cohort=label, epochs=epochs, absolute_delta=diffs.mean(0).tolist(),
                          relative_percent=[float(100 * diffs[:, j].mean() / values[models[0], e, 'ndcg10'][mask].mean()) for j, e in enumerate(epochs)],
                          positive_epochs=int((diffs.mean(0) > 0).sum()),
                          mean_delta_epochs10_15_20=float(diffs[:, 1:].mean())))
    print(json.dumps(paired[-1]), flush=True)

train_support = np.zeros(len(source_ids), dtype=np.int64)
for user in users:
    np.add.at(train_support, np.asarray(user.train[1:]), 1)
data = dict(user_id=uids, target_item_id=targets, train_history_length=lengths,
            visible_context_length=visible, branch_id=branch, target_train_positions=train_support[targets])
changed = np.any(tables[models[1]][0] != tables[models[1]][1], axis=1)
data['target_sid_changed'] = (branch == 1) & changed[targets]
data['context_sid_changed'] = (branch == 1) & np.any(changed[np.maximum(contexts, 0)] & (contexts >= 0), axis=1)
for model in models:
    for epoch in epochs:
        data[model + f'_rank_e{epoch}'] = ranks[model, epoch]
np.savez_compressed(out / 'user-metrics.npz', **data)
write('cohorts', cohort_rows);write('paired-epoch20', paired);write('epoch-stability', stability);write('training', training)
write('verification', dict(source_commit=manifest['git_sha'], exact_config_verified=True,
                           original_and_ALS_mapping_bytes_and_two_table_hashes_verified=True,
                           neural_initialization_and_full_training_stream_verified=True,
                           all8_branch_metadata_lengths_users_targets_and_ranking_metrics_verified=True,
                           invalid_and_duplicate_counts_verified=True, prediction_files=verified,
                           bootstrap_repeats=2000, bootstrap_seed=2748,
                           test_evaluations=0, new_training=False, new_inference=False,
                           caution='Conditional,unadjusted intervals;one fitted seed. History cohorts are different users/targets.'))
for filename, rows in [('cohorts', cohort_rows), ('paired-epoch20', paired)]:
    with (out / (filename + '.csv')).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

lookup = lambda cohort, model, epoch: next(r for r in cohort_rows if r['cohort'] == cohort and r['model'] == model and r['epoch'] == epoch)
lines = ['# One generator with original/ALS SID routing: matched validation', '',
         'Control uses original SID in both branches;experiment uses ALS SID at10+observed events. '
         'Both have the same SHORT/LONGprefix token. Primary endpoint20epochs. '
         'All saved beams were independently decoded with recomputed per-row branch IDs.', '',
         '| Epoch | Control NDCG10 | Gated NDCG10 | Relative change |', '|---|---:|---:|---:|']
for e in epochs:
    a, b = [lookup('all', m, e)['ndcg10'] for m in models]
    lines.append(f'| {e} | {a:.6f} | {b:.6f} | {100*(b/a-1):+.2f}% |')
lines += ['', '| TRAIN events | Users | Control20 | Gated20 | Change | User95% delta interval |',
          '|---|---:|---:|---:|---:|---|']
for r in paired:
    lines.append(f"| {r['cohort']} | {r['users']} | {r['original']:.6f} | {r['gated']:.6f} | {r['relative_percent']:+.2f}% | [{r['user_ci95'][0]:+.6f},{r['user_ci95'][1]:+.6f}] |")
lines += ['', 'The all/<10/10+rows overlap the finer length bins. Model sees at most20events. '
          'User and target-cluster2000-replicate intervals condition on fitted weights, '
          'are unadjusted,and do not capture seed variability. Epoch checks share users '
          'and trajectories. Neither equality nor superiority is established by similar point estimates.', '',
          'Both models completed20epochs,2628260examples and10280updates,zero AMPskips. '
          'Body1755.8seconds. Same4978176parameters,initialization,raw examples and order. '
          'Four best/last checkpoint files remain on private Kaggle;no weights were downloaded.', '',
          'Short-branch input/label SIDassignments agree between models,but predictions may '
          'change through shared weights trained on different LONGlabels/inputs. Therefore '
          'the gate does not freeze short-user quality at its control value.', '',
          'The earlier unconditioned original at20scored0.036943overall,above both current '
          'fits. On20+it scored0.039276,pureALS0.044327;current tagged control0.046972 '
          'and gated model0.044165. These are historical fitted references,not a matched '
          'ablation identifying the branch token or shared-model cost.', '',
          'Files:cohorts.json/csv(all4metrics/4epochs),paired-epoch20.json/csv(user and '
          'target-cluster intervals),epoch-stability.json,training.json,verification.json '
          'and user-metrics.npz(all8ranks,branch,lengths,target support/path-change flags). '
          'Rank999means absent from final20beams. Raw cache:/tmp/rhpr-gated-index-completed. '
          'Authoritative source:private carasik/rhpr-gated-sid-tiger-screen,version1. '
          'No new GPUjob,inference,test or threshold tuning was performed.']
(out / 'README.md').write_text('\n'.join(lines) + '\n')

# Check the cases that actually change SIDassignments. Similar aggregate quality
# is weak evidence of learning two mappings when most assignments are identical.
subsets = dict(long_target_changed=data['target_sid_changed'],
               long_context_changed=data['context_sid_changed'],
               long_any_input_or_target_changed=(branch == 1) & (data['target_sid_changed'] | data['context_sid_changed']),
               long_no_input_or_target_change=(branch == 1) & ~data['target_sid_changed'] & ~data['context_sid_changed'])
structural, aliases = [], []
for label, mask in subsets.items():
    row = dict(cohort=label, users=int(mask.sum()), unique_targets=int(np.unique(targets[mask]).size))
    for model in models:
        r = ranks[model, 20][mask].astype(float)
        row[model] = dict(ndcg10=float(np.where(r <= 10, 1/np.log2(r + 1), 0).mean()), hits10=int((r <= 10).sum()))
    structural.append(row)
mask = data['target_sid_changed']
for epoch in epochs:
    for model in models:
        with np.load(root / 'trials' / model / f'validation-epoch-{epoch:03d}.npz') as f:
            predictions = f['semantic_ids'][mask]
            hits = []
            for table in tables[models[1]]:
                equal = np.all(predictions == table[targets[mask], None, :], axis=-1)
                r = np.where(equal.any(1), equal.argmax(1) + 1, 999)
                hits.append(r <= 10)
        aliases.append(dict(epoch=epoch, model=model, users=int(mask.sum()),
                            original_sid_hits10=int(hits[0].sum()), als_sid_hits10=int(hits[1].sum()),
                            both=int((hits[0] & hits[1]).sum())))
write('structural-subsets', dict(subsets=structural, alias_hits=aliases,
                                caveat='Post-hoc small cohorts. The other mapping SID names a different item in the active mapping;counts do not identify ignored-gate behavior. Shared-weight effects can reach unchanged paths.'))
with (out / 'README.md').open('a') as f:
    f.write('\n\nStructural support:only116validation targets(74distinct items)are in the LONGbranch '
            'and have different target SIDs. At20the control has7top10hits there,the gated '
            'model4. This is too small to establish a mapping-specific quality effect. '
            'structural-subsets.json retains these counts,changed-context cohorts and '
            'cross-mapping SIDbeam coverage across all epochs. Subset rows overlap. '
            'These two mappings agree on96%of catalogue items.\n')
if args.plot:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for ax, label in zip(axes.flat[:3], ['all', '<10', '10+']):
        for model, title, color in zip(models, ['Original+branch token', 'Gated original/ALS'], ['black', '#d57912']):
            ax.plot(epochs, [lookup(label, model, e)['ndcg10'] for e in epochs], marker='o', label=title, color=color)
        ax.set(title=f"{label} TRAIN events (n={lookup(label, models[0], 20)['users']:,})", xlabel='Epoch', ylabel='Validation NDCG@10')
        ax.set_xticks(epochs);ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    ax = axes[1, 1]
    rows = [r for r in paired if r['cohort'] in ['3', '4', '5-9', '10-19', '20+']]
    means = np.array([r['delta'] for r in rows]);ci = np.array([r['user_ci95'] for r in rows])
    ax.errorbar(np.arange(len(rows)), means, yerr=[means-ci[:, 0], ci[:, 1]-means], fmt='o', capsize=4, color='#d57912')
    ax.axhline(0, color='black', linewidth=.8);ax.axvline(2.5, color='gray', linestyle='--', alpha=.5)
    ax.set_xticks(np.arange(len(rows)), [r['cohort'] for r in rows])
    ax.set(title='Epoch20: paired user95% intervals', xlabel='TRAIN events; ALS starts at10', ylabel='Gated minus control NDCG@10')
    ax.grid(alpha=.2)
    fig.savefig(out / 'validation-and-length.png', dpi=140)
    plt.close(fig)
