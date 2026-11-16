"""Replay the completed Steam length pilot and conditional paired user intervals."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from rhpr.gr.movie_data import case_digest
from rhpr.gr.sid import SidIndex, evaluate_ranked_sids

SOURCE = '72ceff63e025cb1cb4731672b665594c6f97db3b'
REPLICATION_SOURCE = 'a16145d4aa6e3ee526198482712d0282d7a8bc59'


def read(p): return json.loads(Path(p).read_text())
def write(p, v): Path(p).write_text(json.dumps(v, indent=2, allow_nan=False)+'\n')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def analyze(root, data, out):
    root, data, out = map(Path, (root, data, out)); out.mkdir(parents=True, exist_ok=True)
    c = read(root/'experiment.json'); manifest = read(root/'run-manifest.json')
    from rhpr.gr.history_windows import validate
    validate(c)
    source = REPLICATION_SOURCE if c.get('replication') is not None else SOURCE
    windows = c['windows']; train_examples = c['cohorts']['steam']['train_examples']
    suite = read(root/'suite-results.json'); study = read(root/'study-summary.json'); metrics = read(root/'metrics.json')
    assert manifest['git_sha'] == source and manifest['config'] == c
    assert suite['results'] == study['results'] and not suite['failures'] and metrics['models'] == len(windows)
    assert (root/'summary.md').is_file()
    assert set(suite['results']) == {f'steam_recent{w}_42' for w in windows}
    for name, expected in c['pilot_input']['sha256'].items(): assert sha(data/name) == expected
    sidpath = root/'frozen-indexes/steam/items-42.npy'; sid = np.load(sidpath, allow_pickle=False)
    assert sha(sidpath) == c['expected_sid_file_sha256']
    idx = SidIndex.from_full_sids(sid, width=256)
    raw = {}
    for split in ('train', 'validation', 'test'):
        with np.load(root/'prepared-cases/steam'/(split+'-cases.npz')) as z:
            raw[split] = tuple(z[k] for k in ('contexts', 'user_ids', 'targets'))
    assert len(raw['train'][2]) == train_examples
    assert all(len(raw[s][2]) == 2000 and np.all(raw[s][0] >= 0) for s in ('validation', 'test'))
    from rhpr.gr.data import load_beauty, build_examples
    _, users, _, _ = load_beauty(data, c['cohorts']['steam'])
    for split in raw:
        for actual, expected in zip(raw[split], build_examples(users, 30, split)):
            np.testing.assert_array_equal(actual, expected)
    provenance = read(data/'source-provenance.json')
    assert read(root/'cohort-provenance.json') == provenance
    if c.get('replication') is not None:
        selected_ids, excluded_ids = provenance['selected_source_user_ids'], provenance['excluded_source_user_ids']
        assert len(set(selected_ids)) == len(set(excluded_ids)) == 2000 and not set(selected_ids) & set(excluded_ids)
        assert [r['source_sasrec_user_id'] for r in read(data/'user-audit.json')] == selected_ids
    steps = math.ceil(len(raw['train'][2])/512)
    rows = []; curves = {}; credits = {}; selected = {}; prediction_files = invalid = duplicates = candidates = 0
    checkpoints = []

    def replay(path, split, expected):
        nonlocal prediction_files, invalid, duplicates, candidates
        with np.load(path, allow_pickle=False) as z:
            pred = z['semantic_ids']; scores = z['scores']
            np.testing.assert_array_equal(z['user_ids'], raw[split][1])
            np.testing.assert_array_equal(z['target_item_ids'], raw[split][2])
            assert np.isfinite(scores).all() and pred.shape == (2000, 20, 4)
        actual = evaluate_ranked_sids(idx, pred, raw[split][2], cutoffs=(5, 10))
        for key, value in actual.items(): assert abs(value-expected[key]) < 1e-12, (path, key)
        hit = (pred[:, :10] == sid[raw[split][2], None, :]).all(-1)
        credit = np.where(hit.any(1), 1/np.log2(hit.argmax(1)+2), 0.)
        assert abs(float(credit.mean())-actual['ndcg@10']) < 1e-12
        prediction_files += 1; invalid += actual['invalid_predictions']
        duplicates += actual['duplicate_predictions']; candidates += actual['prediction_count']
        return credit

    def lr(step):
        warm, total = 2*steps, 60*steps
        if step < warm: return .003*(step+1)/warm
        return .001+.002*.5*(1+math.cos(math.pi*min(1., (step-warm+1)/(total-warm))))

    for width in windows:
        name = f'steam_recent{width}_42'; trial = suite['results'][name]; p = root/'trials'/name
        assert read(p/'trial-result.json') == trial and trial['sid_file_sha256'] == sha(sidpath)
        assert trial['raw_case_sha256'] == {s: case_digest(raw[s]) for s in ('train', 'validation')}
        assert trial['case_targets_sha256'] == {s: case_digest(raw[s][1:]) for s in ('train', 'validation')}
        assert trial['test_case_sha256'] == case_digest(raw['test'])
        assert trial['initial_state_sha256'] == c['expected_initial_state_sha256']
        curve = read(p/'training-curves.json'); curves[width] = curve; selection = trial['selection']
        stored_selection = read(p/'training-selection.json')
        assert stored_selection['criterion'] == 'validation_item_ndcg@10'
        assert all(stored_selection[k] == v for k, v in selection.items())
        assert [r['epoch'] for r in curve] == list(range(1, selection['training_epochs']+1))
        previous_updates = 0; ema = reference = last = None; evaluated = []
        for r in curve:
            ep = r['epoch']; assert r['updates']+r['amp_skipped_updates'] == steps*ep
            assert r['train_examples'] == len(raw['train'][2])
            assert abs(r['learning_rate']-lr(previous_updates)) < 1e-14
            assert abs(r['next_learning_rate']-lr(r['updates'])) < 1e-14
            previous_updates = r['updates']
            if 'validation' not in r: continue
            assert ep % 5 == 0
            credits[('validation', width, ep)] = replay(p/f'validation-epoch-{ep:03d}.npz', 'validation', r['validation'])
            score = r['validation']['ndcg@10']; ema = score if ema is None else .5*score+.5*ema
            if reference is None or ema > reference+.0005: reference = ema; last = ep
            assert (r['validation_ema_ndcg10'], r['ema_reference_ndcg10'], r['ema_last_improvement_epoch']) == (ema, reference, last)
            evaluated.append(r)
            if ep >= 40 and ep < len(curve): assert ep-last < 20
        assert selection['final_early_stopping_state'] == dict(ema=ema, reference=reference, last_improvement_epoch=last)
        best = max(evaluated, key=lambda r: r['validation']['ndcg@10'])
        assert selection['best_epoch'] == best['epoch'] and selection['validation_ndcg10'] == best['validation']['ndcg@10']
        assert selection['validation_checks'] == len(evaluated) == len(list(p.glob('validation-epoch-*.npz')))
        assert selection['optimizer_updates']+selection['amp_skipped_updates'] == steps*len(curve)
        if selection['stopping_reason'] == 'early_stopping': assert len(curve) >= 40 and len(curve)-last >= 20
        else: assert selection['stopping_reason'] == 'epoch_limit' and len(curve) == 120
        assert abs(sum(r['train_seconds'] for r in curve)-selection['optimizer_training_seconds']) < 1e-6
        assert trial['test']['epoch'] == best['epoch'] and read(p/'test-metrics.json') == trial['test']
        selected[('validation', width)] = credits[('validation', width, best['epoch'])]
        selected[('test', width)] = replay(p/'test-best-predictions.npz', 'test', trial['test']['metrics'])
        rows.append(dict(window=width, best_epoch=best['epoch'], stop_epoch=len(curve), stopping_reason=selection['stopping_reason'],
            validation_ndcg10=selection['validation_ndcg10'], test_ndcg10=trial['test']['metrics']['ndcg@10'],
            validation_hits10=round(best['validation']['recall@10']*2000), test_hits10=round(trial['test']['metrics']['recall@10']*2000),
            test_recall10=trial['test']['metrics']['recall@10'], val_ce_min_epoch=min(evaluated, key=lambda r:r['validation_loss']['total_ce'])['epoch'],
            optimizer_training_seconds=selection['optimizer_training_seconds'], validation_seconds=selection['validation_seconds'],
            warm_epoch_seconds=trial['timing']['warm_epoch_training_median_seconds'], fit_wall_seconds=trial['timing']['wall_seconds'],
            first_epoch_seconds=curve[0]['train_seconds'], amp_skips=selection['amp_skipped_updates'],
            peak_allocated_gib=trial['timing']['cuda_peak_allocated_bytes']/2**30,
            forward_gflops=trial['forward']['supported_forward_flops_batch1']/1e9))
        import torch
        from rhpr.gr.history_windows import encode_window
        from rhpr.gr.generator_resume import array_digest
        assert sha(p/'generator-best.pt') == trial['test']['checkpoint_sha256']
        for kind, epoch in [('best', best['epoch']), ('last', len(curve))]:
            path = p/f'generator-{kind}.pt'; ck = torch.load(path, map_location='cpu', weights_only=False)
            assert ck['epoch'] == epoch and all(torch.isfinite(t).all() for t in ck['model'].values())
            assert ck['config']['training'] == c['training'] and ck['config']['generator'] == c['generator']
            assert ck['config']['active_trial']['id'] == name and ck['config']['data']['history_items'] == width
            if kind == 'last':
                for key in ('optimizer', 'scaler', 'scheduler', 'torch_rng', 'cuda_rng', 'python_rng', 'numpy_rng', 'early_stopping_state'):
                    assert key in ck
                assert ck['best_epoch'] == best['epoch'] and ck['updates'] == selection['optimizer_updates']
                assert ck['early_stopping_state'] == selection['final_early_stopping_state']
                assert ck['sid_matrix_sha256'] == hashlib.sha256(sid.tobytes()).hexdigest()
                for split in ('train', 'validation'):
                    h, u, t = raw[split]
                    assert ck[split+'_case_sha256'] == array_digest((encode_window(h, sid, width), u, t))
                for state in ck['optimizer']['state'].values():
                    assert all(torch.isfinite(v).all() for v in state.values() if torch.is_tensor(v))
            checkpoints.append(dict(window=width, kind=kind, epoch=epoch, sha256=sha(path), bytes=path.stat().st_size, finite=True))
            del ck

    draws = np.random.default_rng(4242).integers(0, 2000, size=(10000, 2000)); comparisons = []
    for split in ('validation', 'test'):
        for left, right in ((30, 20), (30, 10), (20, 10)):
            if left not in windows or right not in windows: continue
            delta = selected[(split, left)]-selected[(split, right)]
            comparisons.append(dict(split=split, left=left, right=right, absolute_delta=float(delta.mean()),
                relative_delta=float(selected[(split, left)].mean()/selected[(split, right)].mean()-1),
                conditional_user_ci95=np.quantile(delta[draws].mean(1), [.025, .975]).tolist(),
                improved=int((delta > 0).sum()), worsened=int((delta < 0).sum()), tied=int((delta == 0).sum())))
    audit = read(data/'user-audit.json'); history = np.array([v['events']-2 for v in audit])
    clean_boundaries = np.array([v['train_last_day'] < v['validation_day'] < v['test_day'] for v in audit])
    frequency = np.bincount(raw['train'][2], minlength=len(sid)); groups = []; user_rows = []
    for split in ('validation', 'test'):
        targets = raw[split][2]; support = frequency[targets]
        masks = [('all', np.ones(2000, dtype=bool)), ('distinct_train_val_test_days', clean_boundaries),
                 ('available_history_30_49', history < 50), ('available_history_50_99', (history >= 50)&(history < 100)),
                 ('available_history_100plus', history >= 100), ('target_train_positions_0', support == 0),
                 ('target_train_positions_1_5', (support >= 1)&(support <= 5)),
                 ('target_train_positions_6_20', (support >= 6)&(support <= 20)), ('target_train_positions_21plus', support > 20)]
        for label, mask in masks:
            groups.append(dict(split=split, group=label, users=int(mask.sum()),
                ndcg10={str(w):float(selected[(split, w)][mask].mean()) if mask.any() else None for w in windows}))
        for j, uid in enumerate(raw[split][1]):
            user_rows.append(dict(split=split, user_id=int(uid), target_item_id=int(targets[j]), available_history=int(history[j])+(split == 'test'),
                distinct_boundary_days=bool(clean_boundaries[j]), target_train_positions=int(support[j]),
                **{f'ndcg10_recent{w}':float(selected[(split, w)][j]) for w in windows}))
    fixed = []
    for ep in (30, 40, 50, 55, 60):
        if all(('validation', w, ep) in credits for w in windows):
            fixed.append(dict(epoch=ep, validation_ndcg10={str(w):float(credits[('validation', w, ep)].mean()) for w in windows}))
    for name, records in [('trial-metrics.csv', rows), ('user-metrics.csv', user_rows)]:
        with (out/name).open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    result = dict(source=source, session_seconds=manifest['duration_seconds'], new_models=len(windows), pipeline_seeds=[42],
        validation_users=2000, test_users=2000, train_examples=train_examples, sid_file_sha256=sha(sidpath),
        data_seed=c['selection']['data_seed'], disjoint_replication=c.get('replication') is not None,
        all_input_cases_metrics_lr_ema_and_selection_verified=True, prediction_files_replayed=prediction_files,
        total_candidates=candidates, invalid_candidates=invalid, duplicate_candidates=duplicates,
        trials=rows, comparisons=comparisons, fixed_epoch_validation=fixed, exploratory_groups=groups,
        checkpoints=checkpoints,
        limits=['One generator/index seed and one selected long-user cohort; reproducibility is not established.',
                'Paired user intervals condition on fitted models and selected validation checkpoints; no seed/item clustering or multiplicity correction.',
                'Within-day event order remains arbitrary; distinct-boundary analysis is descriptive, not a replacement split.',
                'Different windows consume different compute; this is history utility, not a compression/Pareto result.',
                'EMA stopping is an operational rule, not proof of global convergence.'])
    write(out/'analysis.json', result)
    return result, curves


def plot(result, curves, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = {10:'#64748b', 20:'#2563eb', 30:'#15803d'}
    fig, ax = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
    for width in sorted(curves):
        all_rows = curves[width]; rows = [r for r in all_rows if 'validation' in r]
        label = f'Recent{width}'
        ax[0, 0].plot([r['epoch'] for r in rows], [r['validation']['ndcg@10'] for r in rows], color=colors[width], label=label, marker='.')
        ax[0, 1].plot([r['epoch'] for r in rows], [r['validation_loss']['total_ce'] for r in rows], color=colors[width], label=label)
        cumulative = np.cumsum([r['train_seconds'] for r in all_rows])/60
        ax[1, 0].plot([cumulative[r['epoch']-1] for r in rows], [r['validation']['ndcg@10'] for r in rows], color=colors[width], label=label, marker='.')
        ax[1, 1].plot([r['epoch'] for r in all_rows], [r['train_loss_sum_over_codes'] for r in all_rows], color=colors[width], label=label)
    for a, title, xlabel, ylabel in [(ax[0,0], 'Validation ranking', 'Epoch', 'NDCG@10'), (ax[0,1], 'Validation teacher-forced loss', 'Epoch', 'Sum of code CE'),
                                    (ax[1,0], 'Ranking vs optimizer training time', 'Training minutes (excludes validation)', 'NDCG@10'),
                                    (ax[1,1], 'Training loss', 'Epoch', 'Sum of code CE')]:
        a.set(title=title, xlabel=xlabel, ylabel=ylabel); a.grid(alpha=.2)
    ax[0,0].legend(); fig.suptitle(f"Steam long-user cohort data seed{result['data_seed']}: fixed index and generator seed42")
    fig.savefig(out/'learning-curves.png', dpi=180); fig.savefig(out/'learning-curves.pdf'); plt.close(fig)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--results', type=Path, required=True); p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); a = p.parse_args()
    result, curves = analyze(a.results, a.data, a.output); plot(result, curves, a.output)
    print(json.dumps({k: result[k] for k in ('session_seconds', 'trials', 'comparisons', 'prediction_files_replayed')}, indent=2))
