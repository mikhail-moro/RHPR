"""Replay all12 fixed Steam Recent10/20 cells, including three hashed controls."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from rhpr.gr.data import build_examples, load_beauty
from rhpr.gr.generator_resume import array_digest
from rhpr.gr.history_windows import encode_window, validate, verify_matched_results
from rhpr.gr.movie_data import case_digest
from rhpr.gr.sid import SidIndex, evaluate_ranked_sids

SOURCE = 'e710aec495fce54dac88979da910e004bb639854'


def read(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path, value): Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def audit_trial(path, row, source_trial, c, raw, sid, index, counters):
    seed, width = row['seed'], row['history_items']; n = len(raw['validation'][2]); steps = math.ceil(len(raw['train'][2])/512)
    assert row['initial_state_sha256'] == c['expected_initial_state_by_seed'][str(seed)]
    assert row['raw_case_sha256'] == {s: case_digest(raw[s]) for s in ('train', 'validation')}
    assert row['case_targets_sha256'] == {s: case_digest(raw[s][1:]) for s in ('train', 'validation')}
    assert row['test_case_sha256'] == case_digest(raw['test'])
    curve = read(path/'training-curves.json'); selection = row['selection']; stored = read(path/'training-selection.json')
    assert stored['criterion'] == 'validation_item_ndcg@10' and all(stored[k] == v for k, v in selection.items())
    assert [r['epoch'] for r in curve] == list(range(1, selection['training_epochs']+1))
    assert row['forward']['supported_forward_flops_batch1'] == {10:217327232, 20:427087952}[width]

    def replay(file, split, expected):
        with np.load(file, allow_pickle=False) as z:
            pred, scores = z['semantic_ids'], z['scores']
            np.testing.assert_array_equal(z['user_ids'], raw[split][1]); np.testing.assert_array_equal(z['target_item_ids'], raw[split][2])
            assert pred.shape == (n, 20, 4) and np.isfinite(scores).all()
        metrics = evaluate_ranked_sids(index, pred, raw[split][2], cutoffs=(5, 10))
        for k, v in metrics.items(): assert abs(v-expected[k]) < 1e-12, (file, k)
        hit = (pred[:, :10] == sid[raw[split][2], None, :]).all(-1)
        credit = np.where(hit.any(1), 1/np.log2(hit.argmax(1)+2), 0.)
        assert abs(float(credit.mean())-metrics['ndcg@10']) < 1e-12
        counters['files'] += 1
        for field, key in [('candidates', 'prediction_count'), ('invalid', 'invalid_predictions'), ('duplicates', 'duplicate_predictions')]:
            counters[field] += metrics[key]
        return credit

    def lr(step):
        warm, total = 2*steps, 60*steps
        if step < warm: return .003*(step+1)/warm
        return .001+.001*(1+math.cos(math.pi*min(1., (step-warm+1)/(total-warm))))

    ema = reference = last = None; updates = 0; evaluated = []; val_credits = {}
    for r in curve:
        ep = r['epoch']; assert r['updates']+r['amp_skipped_updates'] == steps*ep
        assert r['train_examples'] == len(raw['train'][2])
        assert abs(r['learning_rate']-lr(updates)) < 1e-14 and abs(r['next_learning_rate']-lr(r['updates'])) < 1e-14
        updates = r['updates']
        if 'validation' not in r: continue
        assert ep % 5 == 0
        val_credits[ep] = replay(path/f'validation-epoch-{ep:03d}.npz', 'validation', r['validation'])
        score = r['validation']['ndcg@10']; ema = score if ema is None else .5*score+.5*ema
        if reference is None or ema > reference+.0005: reference = ema; last = ep
        assert (r['validation_ema_ndcg10'], r['ema_reference_ndcg10'], r['ema_last_improvement_epoch']) == (ema, reference, last)
        if ep >= 40 and ep < len(curve): assert ep-last < 20
        evaluated.append(r)
    best = max(evaluated, key=lambda r:r['validation']['ndcg@10'])
    assert best['epoch'] == selection['best_epoch'] and best['validation']['ndcg@10'] == selection['validation_ndcg10']
    assert selection['final_early_stopping_state'] == dict(ema=ema, reference=reference, last_improvement_epoch=last)
    assert selection['optimizer_updates'] == updates and updates+selection['amp_skipped_updates'] == steps*len(curve)
    assert selection['validation_checks'] == len(evaluated) == len(list(path.glob('validation-epoch-*.npz')))
    assert abs(sum(r['train_seconds'] for r in curve)-selection['optimizer_training_seconds']) < 1e-6
    if selection['stopping_reason'] == 'early_stopping': assert len(curve) >= 40 and len(curve)-last >= 20
    else: assert selection['stopping_reason'] == 'epoch_limit' and len(curve) == 120
    assert read(path/'test-metrics.json') == row['test'] and row['test']['epoch'] == best['epoch']
    assert sha(path/'generator-best.pt') == row['test']['checkpoint_sha256']
    selected = dict(validation=val_credits[best['epoch']], test=replay(path/'test-best-predictions.npz', 'test', row['test']['metrics']))
    checkpoints = []
    for kind, epoch in [('best', best['epoch']), ('last', len(curve))]:
        file = path/f'generator-{kind}.pt'; ck = torch.load(file, map_location='cpu', weights_only=False)
        assert ck['epoch'] == epoch and all(torch.isfinite(v).all() for v in ck['model'].values())
        assert ck['config']['seed'] == seed and ck['config']['active_trial']['id'] == source_trial
        assert ck['config']['training'] == c['training'] and ck['config']['generator'] == c['generator']
        assert ck['config']['data']['history_items'] == width
        if kind == 'last':
            for key in ('optimizer', 'scaler', 'scheduler', 'torch_rng', 'cuda_rng', 'python_rng', 'numpy_rng', 'early_stopping_state'): assert key in ck
            assert ck['best_epoch'] == best['epoch'] and ck['updates'] == updates
            assert ck['early_stopping_state'] == selection['final_early_stopping_state']
            assert ck['sid_matrix_sha256'] == hashlib.sha256(sid.tobytes()).hexdigest()
            for split in ('train', 'validation'):
                h, u, t = raw[split]; assert ck[split+'_case_sha256'] == array_digest((encode_window(h, sid, width), u, t))
            for state in ck['optimizer']['state'].values():
                assert all(torch.isfinite(v).all() for v in state.values() if torch.is_tensor(v))
        checkpoints.append(dict(trial=row['trial'], kind=kind, epoch=epoch, sha256=sha(file), bytes=file.stat().st_size, finite=True))
        del ck
    summary = dict(trial=row['trial'], cohort=row['dataset'], seed=seed, window=width,
        validation_ndcg10=selection['validation_ndcg10'], test_ndcg10=row['test']['metrics']['ndcg@10'],
        test_ndcg5=row['test']['metrics']['ndcg@5'], test_recall10=row['test']['metrics']['recall@10'],
        test_hits10=round(n*row['test']['metrics']['recall@10']), best_epoch=best['epoch'], stop_epoch=len(curve),
        stopping_reason=selection['stopping_reason'], val_ce_min_epoch=min(evaluated, key=lambda r:r['validation_loss']['total_ce'])['epoch'],
        warm_epoch_seconds=row['timing']['warm_epoch_training_median_seconds'], fit_wall_seconds=row['timing']['wall_seconds'],
        training_seconds=selection['optimizer_training_seconds'], validation_seconds=selection['validation_seconds'],
        amp_skips=selection['amp_skipped_updates'], forward_gflops=row['forward']['supported_forward_flops_batch1']/1e9)
    return summary, curve, selected, checkpoints


def analyze(artifacts, out):
    artifacts, out = Path(artifacts), Path(out); out.mkdir(parents=True, exist_ok=True)
    root = artifacts/'steam-recent10-20-results'; c = read(root/'experiment.json'); validate(c)
    assert c == read('experiments/steam_recent10_20_crossed_2t4.json')
    manifest = read(root/'run-manifest.json'); suite = read(root/'suite-results.json'); study = read(root/'study-summary.json'); metrics = read(root/'metrics.json')
    assert manifest['git_sha'] == SOURCE and manifest['config'] == c and not suite['failures']
    assert study['results'] == suite['results'] and study['reused_results'] == c['reused_results']
    assert metrics['models'] == 9 and metrics['reused_models'] == 3 and metrics['total_comparison_models'] == 12
    assert (root/'summary.md').is_file(); verify_matched_results(c, suite['results'])
    stems = {'steam_a':'steam-history', 'steam_b':'steam-history-replica'}
    raw = {}; audits = {}; source_users = {}; array_hashes = {}; catalog = None
    for cohort, stem in stems.items():
        data = artifacts/(stem+'-dataset')
        for name, expected in c['cohort_inputs'][cohort]['sha256'].items(): assert sha(data/name) == expected
        texts, users, _, _ = load_beauty(data, c['cohorts'][cohort])
        catalogs = read(data/'catalog.json')
        if catalog is None: catalog = catalogs
        else: assert catalog == catalogs
        assert read(root/'cohort-provenance.json')[cohort] == read(data/'source-provenance.json')
        raw[cohort] = {}
        for split in ('train', 'validation', 'test'):
            with np.load(root/'prepared-cases'/cohort/(split+'-cases.npz'), allow_pickle=False) as z:
                raw[cohort][split] = tuple(z[k] for k in ('contexts', 'user_ids', 'targets'))
            for actual, expected in zip(raw[cohort][split], build_examples(users, 30, split)): np.testing.assert_array_equal(actual, expected)
        audits[cohort] = read(data/'user-audit.json'); source_users[cohort] = {r['source_sasrec_user_id'] for r in audits[cohort]}
        array_hashes[cohort] = {}
        for file in ('items-42.npy', 'item-embeddings.npy', 'centroids-42.npy'):
            path = root/'frozen-indexes'/cohort/file; digest = sha(path)
            assert digest == sha(artifacts/'steam-history-results/frozen-indexes/steam'/file)
            array_hashes[cohort][file] = digest
    assert not source_users['steam_a'] & source_users['steam_b']
    sidpath = root/'frozen-indexes/steam_a/items-42.npy'; assert sha(sidpath) == c['expected_sid_file_sha256']
    sid = np.load(sidpath, allow_pickle=False); index = SidIndex.from_full_sids(sid, width=256)
    records = dict(suite['results']); locations = {name:(root/'trials'/name, name) for name in records}
    for name, reference in c['reused_results'].items():
        cohort = reference['result']['dataset']; oldroot = artifacts/(stems[cohort]+'-results'); path = oldroot/'trials'/reference['source_trial']
        assert sha(path/'trial-result.json') == reference['source_result_sha256'] and sha(path/'generator-best.pt') == reference['best_checkpoint_sha256']
        assert read(oldroot/'run-manifest.json')['git_sha'] == reference['source_sha']
        original = read(path/'trial-result.json'); original.update(trial=name, dataset=cohort); assert original == reference['result']
        records[name] = original; locations[name] = (path, reference['source_trial'])
    counters = dict(files=0, candidates=0, invalid=0, duplicates=0); rows = []; curves = {}; selected = {}; checkpoints = []
    for name, row in sorted(records.items()):
        path, source_trial = locations[name]
        if name in suite['results']:
            assert read(path/'trial-result.json') == row
            assert metrics[name+'_validation_ndcg10'] == row['selection']['validation_ndcg10']
            assert metrics[name+'_test_ndcg10'] == row['test']['metrics']['ndcg@10']
        summary, curve, credits, ck = audit_trial(path, row, source_trial, c, raw[row['dataset']], sid, index, counters)
        summary['reused'] = name in c['reused_results']; rows.append(summary); curves[name] = curve; checkpoints.extend(ck)
        for split, values in credits.items(): selected[row['dataset'], row['seed'], row['history_items'], split] = values
    rng = np.random.default_rng(4242); draws = {k:rng.integers(0, 2000, size=(10000, 2000)) for k in stems}
    pairs = []; averages = []; groups = []; items = []; user_rows = []
    for split in ('validation', 'test'):
        boot_by_cohort = {}; mean_by_cohort = {}
        for cohort in stems:
            a = np.stack([selected[cohort, s, 10, split] for s in c['seeds']]); b = np.stack([selected[cohort, s, 20, split] for s in c['seeds']]); delta = b-a
            for j, seed in enumerate(c['seeds']):
                d = delta[j]
                pairs.append(dict(cohort=cohort, seed=seed, split=split, recent10=float(a[j].mean()), recent20=float(b[j].mean()),
                    absolute_delta=float(d.mean()), relative_delta=float(b[j].mean()/a[j].mean()-1),
                    conditional_user_ci95=np.quantile(d[draws[cohort]].mean(1), [.025, .975]).tolist(),
                    improved=int((d>0).sum()), worsened=int((d<0).sum()), tied=int((d==0).sum())))
            mean_delta = delta.mean(0); boots = mean_delta[draws[cohort]].mean(1); boot_by_cohort[cohort] = boots; mean_by_cohort[cohort] = (a.mean(), b.mean())
            averages.append(dict(cohort=cohort, split=split, recent10=float(a.mean()), recent20=float(b.mean()), absolute_delta=float(mean_delta.mean()),
                relative_delta=float(b.mean()/a.mean()-1), conditional_user_ci95=np.quantile(boots, [.025, .975]).tolist(),
                recent10_seed_sd=float(a.mean(1).std(ddof=1)), recent20_seed_sd=float(b.mean(1).std(ddof=1)),
                paired_delta_seed_sd=float(delta.mean(1).std(ddof=1)), positive_seeds=int((delta.mean(1)>0).sum())))
            targets = raw[cohort][split][2]; support = np.bincount(raw[cohort]['train'][2], minlength=len(sid))[targets]
            history = np.array([r['events']-2 for r in audits[cohort]])
            clean = np.array([r['train_last_day']<r['validation_day']<r['test_day'] for r in audits[cohort]])
            masks = [('all', np.ones(2000, bool)), ('distinct_boundary_days', clean), ('history30_49', history<50),
                ('history50_99', (history>=50)&(history<100)), ('history100plus', history>=100), ('support0', support==0),
                ('support1_5', (support>=1)&(support<=5)), ('support6_20', (support>=6)&(support<=20)), ('support21plus', support>20),
                ('excluding_gta_v_and_fallout4', ~np.isin(targets, [39, 3087]))]
            for label, mask in masks:
                if mask.any(): groups.append(dict(cohort=cohort, split=split, group=label, users=int(mask.sum()), recent10=float(a[:, mask].mean()),
                    recent20=float(b[:, mask].mean()), absolute_delta=float(mean_delta[mask].mean())))
            item_contribution = np.bincount(targets, weights=mean_delta, minlength=len(sid))/2000
            item_counts = np.bincount(targets, minlength=len(sid))
            for item in np.argsort(np.abs(item_contribution))[::-1][:20]:
                items.append(dict(cohort=cohort, split=split, item_id=int(item), title=catalog[item]['title'], targets=int(item_counts[item]),
                    contribution_to_mean_delta=float(item_contribution[item])))
            for j, seed in enumerate(c['seeds']):
                for at, user in enumerate(raw[cohort][split][1]): user_rows.append(dict(cohort=cohort, split=split, seed=seed, user_id=int(user),
                    target_item_id=int(targets[at]), support=int(support[at]), recent10=float(a[j, at]), recent20=float(b[j, at]), delta=float(delta[j, at])))
        mean10 = float(np.mean([v[0] for v in mean_by_cohort.values()])); mean20 = float(np.mean([v[1] for v in mean_by_cohort.values()]))
        averages.append(dict(cohort='both_equal_weight', split=split, recent10=mean10, recent20=mean20, absolute_delta=mean20-mean10,
            relative_delta=mean20/mean10-1, conditional_user_ci95=np.quantile(np.mean(list(boot_by_cohort.values()), axis=0), [.025, .975]).tolist(),
            note='Same users resampled jointly across the three fixed seeds; cohorts resampled independently. Not12000 independent users.'))
    for file, values in [('trial-metrics.csv', rows), ('user-metrics.csv', user_rows), ('pair-metrics.csv', pairs)]:
        with (out/file).open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(values[0])); writer.writeheader(); writer.writerows(values)
    result = dict(source=SOURCE, session_seconds=manifest['duration_seconds'], new_models=9, reused_models=3,
        unique_users=4000, cohorts=list(stems), generator_seeds=c['seeds'], tokenizer_seed=42, windows=[10,20],
        all_input_cases_metrics_lr_ema_selection_and_checkpoints_verified=True, array_hashes=array_hashes,
        replay=counters, trials=rows, paired_comparisons=pairs, averages=averages, exploratory_groups=groups,
        exploratory_item_contributions=items, checkpoints=checkpoints,
        limits=['Two previously fixed long-user cohorts,three generator seeds and one index; no dataset/index-seed replication.',
            'User intervals condition on the12fitted selected models; exclude seed/item clustering/selection uncertainty and multiplicity adjustment.',
            'Five of six paired wins would not be six independent trials; the same users and seeds repeat.',
            'Subgroups and item attribution are descriptive; no new filtering or routing rules.',
            'EMA stopping is an operational rule,not a convergence proof. Different windows consume different compute.'])
    write(out/'analysis.json', result); return result, curves


def plot(result, curves, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for loss in (False, True):
        fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
        for row, cohort in enumerate(result['cohorts']):
            for col, seed in enumerate(result['generator_seeds']):
                ax = axes[row, col]
                for width, color in [(10,'#64748b'),(20,'#2563eb')]:
                    records = [r for r in curves[f'{cohort}_recent{width}_{seed}'] if 'validation' in r]
                    ax.plot([r['epoch'] for r in records], [r['validation_loss']['total_ce'] if loss else r['validation']['ndcg@10'] for r in records],
                        label=f'Recent{width}', color=color, marker='.' if not loss else None)
                ax.set(title=f'{cohort}, generator seed{seed}', xlabel='Epoch', ylabel='Validation code CE' if loss else 'Validation NDCG@10'); ax.grid(alpha=.2)
        axes[0,0].legend(); fig.suptitle('Steam: fixed index, two cohorts, three seeds')
        name = 'validation-loss-curves' if loss else 'learning-curves'; fig.savefig(out/(name+'.png'), dpi=170); fig.savefig(out/(name+'.pdf')); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for ax, split in zip(axes, ['validation','test']):
        values = [r for r in result['paired_comparisons'] if r['split']==split]
        for j, r in enumerate(values):
            lo, hi = r['conditional_user_ci95']; d = r['absolute_delta']
            ax.errorbar(d, j, xerr=[[d-lo],[hi-d]], fmt='o', color='#2563eb')
        ax.set_yticks(range(len(values)), [r['cohort']+' /'+str(r['seed']) for r in values]); ax.invert_yaxis()
        ax.axvline(0,color='#64748b',linewidth=1); ax.grid(axis='x',alpha=.2); ax.set(title=split.capitalize(),xlabel='NDCG@10 difference: Recent20 minus Recent10')
    fig.suptitle('Conditional95% paired-user intervals; generator seeds are fixed')
    fig.savefig(out/'paired-differences.png',dpi=170); fig.savefig(out/'paired-differences.pdf'); plt.close(fig)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--artifacts',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    result, curves = analyze(a.artifacts,a.output); plot(result,curves,a.output)
    print(json.dumps({k:result[k] for k in ('session_seconds','paired_comparisons','averages','replay')},indent=2))
