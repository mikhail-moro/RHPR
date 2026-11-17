"""Replay the five completed ML1M Last-8 windows without model inference."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from rhpr.gr.data import file_sha256
from rhpr.gr.generator_resume import array_digest
from rhpr.gr.ml100k_pair_raw import cases
from rhpr.gr.ml1m_last8 import REFERENCE_SID_SHA256, trial_config
from rhpr.gr.movie_data import case_digest
from rhpr.gr.sid import SidIndex, evaluate_ranked_sids

WINDOWS = (8, 16, 32, 64, 128)


def read(path): return json.loads(Path(path).read_text())
def write(path, value): Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def contrast(a, b):
    delta = np.asarray(a)-np.asarray(b)
    values, counts = np.unique(delta, return_counts=True)
    rng = np.random.default_rng(4242)
    means = np.concatenate([(rng.multinomial(len(delta), counts/len(delta), size=100) @ values)/len(delta)
                            for _ in range(100)])
    return dict(n=len(delta), a=float(np.mean(a)), b=float(np.mean(b)), delta=float(delta.mean()),
        relative_delta=float(np.mean(a)/np.mean(b)-1) if np.mean(b) else None,
        conditional_user_ci95=np.quantile(means, [.025, .975]).tolist(),
        improved=int((delta > 0).sum()), worsened=int((delta < 0).sum()), tied=int((delta == 0).sum()))


def replay(path, split, expected, raw, sid, index, counters):
    with np.load(path) as z:
        pred, scores = z['semantic_ids'], z['scores']
        np.testing.assert_array_equal(z['user_ids'], raw[split][1])
        np.testing.assert_array_equal(z['target_item_ids'], raw[split][2])
    assert pred.shape == (6040, 20, 4) and np.isfinite(scores).all()
    assert np.all(np.diff(scores, axis=1) <= 1e-6)
    actual = evaluate_ranked_sids(index, pred, raw[split][2], cutoffs=(5, 10))
    for key, value in actual.items(): assert abs(value-expected[key]) < 1e-12, (path, key)
    hit = (pred == sid[raw[split][2], None, :]).all(-1)
    ranks = np.where(hit.any(1), hit.argmax(1)+1, 21)
    credit = np.where(ranks <= 10, 1/np.log2(ranks+1), 0.)
    assert abs(credit.mean()-actual['ndcg@10']) < 1e-12
    counters['files'] += 1
    for field, key in [('paths', 'prediction_count'), ('invalid', 'invalid_predictions'), ('duplicates', 'duplicate_predictions')]:
        counters[field] += actual[key]
    return credit, ranks


def audit_trial(root, width, raw, sid, index, counters):
    c = read(root/'experiment.json'); path = root/f'trials/recent{width}_42'
    row = read(path/'trial-result.json'); selection = row['selection']; curve = read(path/'training-curves.json')
    assert row['history_items'] == width and row['seed'] == 42
    assert row['initial_state_sha256'] == c['expected_initial_state_sha256']
    assert row['sid_file_sha256'] == REFERENCE_SID_SHA256
    assert row['raw_case_sha256'] == {s:case_digest(raw[s]) for s in ('train', 'validation')}
    assert row['case_targets_sha256'] == {s:case_digest(raw[s][1:]) for s in ('train', 'validation')}
    assert row['test_case_sha256'] == case_digest(raw['test'])
    assert [r['epoch'] for r in curve] == list(range(1, selection['training_epochs']+1))
    stored = read(path/'training-selection.json')
    assert all(stored[k] == v for k, v in selection.items())
    updates = 0; ema = reference = last = None; evaluations = []; selected = {}
    def lr(step):
        if step < 190: return .003*(step+1)/190
        return .001+.001*(1+math.cos(math.pi*min(1., (step-190+1)/5510)))
    for r in curve:
        ep = r['epoch']
        assert r['train_examples'] == 48320 and r['updates']+r['amp_skipped_updates'] == 95*ep
        assert abs(r['learning_rate']-lr(updates)) < 1e-14
        assert abs(r['next_learning_rate']-lr(r['updates'])) < 1e-14
        updates = r['updates']
        if 'validation' not in r: continue
        assert ep % 5 == 0
        loss = r['validation_loss']
        assert loss['examples'] == 6040 and loss['precision'] == 'fp32'
        assert abs(sum(loss['ce_by_level'])-loss['total_ce']) < 1e-6
        credit = replay(path/f'validation-epoch-{ep:03d}.npz', 'validation', r['validation'], raw, sid, index, counters)
        if ep == selection['best_epoch']: selected['validation'] = credit
        score = r['validation']['ndcg@10']; ema = score if ema is None else .5*score+.5*ema
        if reference is None or ema > reference+.0005: reference = ema; last = ep
        assert (r['validation_ema_ndcg10'], r['ema_reference_ndcg10'], r['ema_last_improvement_epoch']) == (ema, reference, last)
        if ep >= 40 and ep < len(curve): assert ep-last < 20
        evaluations.append(r)
    best = max(evaluations, key=lambda r:r['validation']['ndcg@10'])
    assert best['epoch'] == selection['best_epoch'] and best['validation']['ndcg@10'] == selection['validation_ndcg10']
    assert selection['final_early_stopping_state'] == dict(ema=ema, reference=reference, last_improvement_epoch=last)
    assert selection['validation_checks'] == len(evaluations) == len(list(path.glob('validation-epoch-*.npz')))
    assert selection['optimizer_updates'] == updates and selection['amp_skipped_updates'] == curve[-1]['amp_skipped_updates']
    assert abs(sum(r['train_seconds'] for r in curve)-selection['optimizer_training_seconds']) < 1e-6
    assert selection['stopping_reason'] == 'early_stopping' and len(curve) >= 40 and len(curve)-last >= 20
    assert not row['budget_censored'] and read(path/'test-metrics.json') == row['test']
    assert row['test']['epoch'] == best['epoch'] and not row['test']['seen_items_filtered']
    selected['test'] = replay(path/'test-best-predictions.npz', 'test', row['test']['metrics'], raw, sid, index, counters)
    trial = dict(id=f'recent{width}_42', history_items=width, seed=42); active = trial_config(c, trial)
    checkpoints = []
    for kind, epoch in [('best', best['epoch']), ('last', len(curve))]:
        file = path/f'generator-{kind}.pt'; ck = torch.load(file, map_location='cpu', weights_only=False)
        assert ck['epoch'] == epoch and all(torch.isfinite(v).all() for v in ck['model'].values() if v.is_floating_point())
        assert ck['config']['active_trial'] == trial
        for key in ('seed', 'data', 'generator', 'training'): assert ck['config'][key] == active[key]
        if kind == 'last':
            for key in ('optimizer','scaler','scheduler','torch_rng','cuda_rng','python_rng','numpy_rng'): assert key in ck
            assert ck['updates'] == updates and ck['early_stopping_state'] == selection['final_early_stopping_state']
            assert ck['best_epoch'] == best['epoch'] and ck['sid_matrix_sha256'] == hashlib.sha256(sid.tobytes()).hexdigest()
            for split in ('train', 'validation'):
                h, u, t = raw[split]
                assert ck[split+'_case_sha256'] == array_digest((h[:, -width:].copy(), u, t))
            for state in ck['optimizer']['state'].values():
                assert all(torch.isfinite(v).all() for v in state.values() if torch.is_tensor(v))
        checkpoints.append(dict(kind=kind, epoch=epoch, sha256=file_sha256(file), bytes=file.stat().st_size))
        del ck
    assert row['test']['checkpoint_sha256'] == checkpoints[0]['sha256']
    summary = dict(window=width, validation_ndcg10=selection['validation_ndcg10'], test_ndcg10=row['test']['metrics']['ndcg@10'],
        test_recall10=row['test']['metrics']['recall@10'], best_epoch=best['epoch'], stop_epoch=len(curve),
        stopping_reason=selection['stopping_reason'], warm_epoch_seconds=row['timing']['warm_epoch_training_median_seconds'],
        train_and_validation_seconds=selection['training_seconds'], training_seconds=selection['optimizer_training_seconds'],
        validation_seconds=selection['validation_seconds'], first_epoch_seconds=curve[0]['train_seconds'],
        test_beam_seconds=row['test']['metrics']['seconds'], evaluation_batch=active['training']['evaluation_batch_size'],
        parameters=row['forward']['parameters'], forward_gflops=row['forward']['supported_forward_flops_batch1']/1e9,
        encoder_positions=row['forward']['encoder_positions'], amp_skips=selection['amp_skipped_updates'],
        min_validation_ce=min(r['validation_loss']['total_ce'] for r in evaluations),
        min_ce_epoch=min(evaluations, key=lambda r:r['validation_loss']['total_ce'])['epoch'],
        last_validation_ce=evaluations[-1]['validation_loss']['total_ce'], checkpoints=checkpoints)
    return summary, selected, curve


def analyze(old, new, out):
    out.mkdir(parents=True, exist_ok=True); torch.set_num_threads(1)
    oldc, newc = read(old/'experiment.json'), read(new/'experiment.json')
    for key in ('data','generator','training','sampling','quantizer','expected_initial_state_sha256','expected_raw_case_sha256'):
        assert oldc[key] == newc[key], key
    manifest = read(new/'run-manifest.json'); assert manifest['git_sha'] == 'eceafc4f9217e8a2f79c6fc2686936c80d542487'
    assert manifest['config'] == newc
    suite = read(new/'suite-results.json'); assert not suite['failures'] and len(suite['results']) == 2
    metrics = read(new/'metrics.json'); assert metrics['models'] == 2 and metrics['test_evaluated']
    for file in ('items-42.npy','centroids-42.npy','item-embeddings.npy'):
        assert file_sha256(old/'frozen-indexes'/file) == file_sha256(new/'frozen-indexes'/file), file
    assert file_sha256(new/'frozen-indexes/items-42.npy') == REFERENCE_SID_SHA256
    raw = {s:cases(old/'prepared-cases', s) for s in ('train','validation','test')}
    for split in raw:
        assert case_digest(raw[split]) == oldc['expected_raw_case_sha256'][split]
        assert case_digest(cases(new/'prepared-cases', split)) == case_digest(raw[split])
    sid = np.load(old/'frozen-indexes/items-42.npy'); index = SidIndex.from_full_sids(sid, width=256)
    counters = dict(files=0, paths=0, invalid=0, duplicates=0); rows = {}; credits = {}; curves = {}
    for w in WINDOWS:
        root = new if w in (16,64) else old
        row, selected, curve = audit_trial(root, w, raw, sid, index, counters)
        rows[str(w)] = row; curves[w] = curve
        for split, (credit, ranks) in selected.items(): credits[split, w] = credit
        np.savez_compressed(out/f'selected-ranks-{w}.npz', user_ids=raw['test'][1],
            validation_target_ids=raw['validation'][2], test_target_ids=raw['test'][2],
            validation_ranks=selected['validation'][1], test_ranks=selected['test'][1])
        if w in (16,64):
            assert metrics[f'recent{w}_42_validation_ndcg10'] == row['validation_ndcg10']
            assert metrics[f'recent{w}_42_test_ndcg10'] == row['test_ndcg10']
    contrasts = {s:{f'{a}_vs_{b}':contrast(credits[s,a], credits[s,b])
                   for a,b in [(16,8),(32,8),(64,8),(128,8),(128,64)]} for s in ('validation','test')}
    anchors = {}
    for s in ('validation','test'):
        with np.load(old/'prepared-cases'/f'{s}-anchors.npz') as z: anchors[s] = {k:z[k] for k in z.files}
        np.testing.assert_array_equal(anchors[s]['user_ids'], raw[s][1])
    strict = ((anchors['validation']['target_timestamps'] > anchors['validation']['preceding_timestamps']) &
              (anchors['test']['target_timestamps'] > anchors['test']['preceding_timestamps']))
    strata = {}
    for s in ('validation','test'):
        a = anchors[s]; pos = a['target_positions']; groups = {'strict_both_holdout_boundaries':strict,
            'own_target_after_previous_timestamp':a['target_timestamps'] > a['preceding_timestamps'],
            'own_target_tied_with_previous_timestamp':a['target_timestamps'] == a['preceding_timestamps']}
        for lo,hi in ((0,32),(32,64),(64,128),(128,100000)):
            groups[f'past_events_{lo}_{hi-1}'] = (pos >= lo)&(pos < hi)
        strata[s] = {label:dict(n=int(mask.sum()), mean_ndcg10={str(w):float(credits[s,w][mask].mean()) for w in WINDOWS},
            contrast_128_vs_8=contrast(credits[s,128][mask], credits[s,8][mask]),
            contrast_128_vs_64=contrast(credits[s,128][mask], credits[s,64][mask])) for label,mask in groups.items() if mask.any()}
    flat = [{k:v for k,v in r.items() if k != 'checkpoints'} for r in rows.values()]
    with (out/'model-metrics.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(flat[0])); writer.writeheader(); writer.writerows(flat)
    with (out/'learning-curves.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=['window','epoch','training_ce','validation_ndcg10','validation_ce','lr'])
        writer.writeheader()
        for w,curve in curves.items():
            for r in curve: writer.writerow(dict(window=w,epoch=r['epoch'],training_ce=r['train_loss_sum_over_codes'],
                validation_ndcg10=r.get('validation',{}).get('ndcg@10'),validation_ce=r.get('validation_loss',{}).get('total_ce'),lr=r['learning_rate']))
    result = dict(models=rows, contrasts=contrasts, exploratory_strata=strata, replay=counters,
        shared_sid_file_sha256=REFERENCE_SID_SHA256, shared_raw_case_sha256=oldc['expected_raw_case_sha256'],
        remaining_run_seconds=manifest['duration_seconds'], completed_windows=list(WINDOWS), excluded_window=256,
        index_embeddings_centroids_identical=True, all_ema_stopped=True, additional_gpu_inference=False,
        caveats=['One generator and tokenizer seed; no between-seed uncertainty.',
                 'Bootstrap resamples users conditional on these selected checkpoints; no item clustering or multiplicity correction.',
                 'Length/timestamp strata are exploratory composition comparisons, not causal interventions.',
                 'MovieLens ratings/timestamp tie ordering are not verified viewing order.',
                 'Microbatch size/dropout stream and USER-token distance change with maximum window.',
                 'FLOPs are supported B1 teacher-forced forward operations, not beam serving or training cost.'])
    write(out/'analysis.json', result)
    plot(rows, curves, out)
    return result


def plot(rows, curves, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
    colors=plt.cm.viridis(np.linspace(.05,.9,5))
    for split,marker in [('validation','o'),('test','s')]:
        axes[0,0].plot(WINDOWS,[rows[str(w)][split+'_ndcg10'] for w in WINDOWS],marker=marker,label=split)
    axes[0,0].set(xscale='log',xticks=WINDOWS,xticklabels=WINDOWS,xlabel='History items',ylabel='NDCG@10',title='Selected checkpoints (VAL selection)');axes[0,0].legend()
    for w,col in zip(WINDOWS,colors):
        v=[r for r in curves[w] if 'validation' in r]
        axes[0,1].plot([r['epoch'] for r in v],[r['validation']['ndcg@10'] for r in v],color=col,label=str(w))
        r=rows[str(w)];axes[0,1].scatter(r['best_epoch'],r['validation_ndcg10'],color=col)
        axes[1,0].scatter(r['forward_gflops'],r['test_ndcg10'],color=col)
        axes[1,0].annotate(str(w),(r['forward_gflops'],r['test_ndcg10']),xytext=(4,4),textcoords='offset points')
    axes[0,1].set(xlabel='Epoch',ylabel='VAL NDCG@10',title='Early stopping curves');axes[0,1].legend(title='Items',ncol=3)
    axes[1,0].set(xscale='log',xlabel='Supported B1 forward GFLOPs (not beam)',ylabel='TEST NDCG@10',title='Quality and forward cost')
    axes[1,1].bar([str(w) for w in WINDOWS],[rows[str(w)]['warm_epoch_seconds'] for w in WINDOWS],color=colors)
    axes[1,1].set(xlabel='History items',ylabel='Median warm training epoch, seconds',title='48,320 examples per epoch; T4 / FP16 / SDPA')
    for ax in axes.flat: ax.grid(alpha=.2)
    fig.suptitle('ML1M Last-8: fixed targets, frozen ST5/RQ-KMeans, TIGER 4+4 / d128, seed42')
    for ext in ('png','pdf'): fig.savefig(out/f'length-quality-compute.{ext}',dpi=170)
    plt.close(fig)


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--old',type=Path,default=Path('/tmp/ml1m-last8-stopped-v2'))
    p.add_argument('--new',type=Path,default=Path('/tmp/ml1m-last8-remaining-results'))
    p.add_argument('--output',type=Path,default=Path('/tmp/ml1m-last8-analysis'))
    args=p.parse_args();r=analyze(args.old,args.new,args.output)
    print(json.dumps(dict(models=r['models'],contrasts=r['contrasts'],replay=r['replay']),indent=2))
