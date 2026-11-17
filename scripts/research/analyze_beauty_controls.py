"""CPU-only replay of Beauty Recent12 and the two 6+6/dropout.2 controls."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from rhpr.gr.beauty_pairs import encode, validate
from rhpr.gr.data import build_examples, file_sha256, load_beauty
from rhpr.gr.generator_resume import array_digest
from rhpr.gr.ml100k_pair_raw import cases
from rhpr.gr.movie_data import case_digest
from rhpr.gr.sid import SidIndex, evaluate_ranked_sids

SOURCES = {'recent12': '413f2afb993b5b56f93534a7ccef155313e89938',
           'mender-depth': 'a68b70dbfe15b386d40552923afbe9801a7f5b90'}
OLD_SOURCE = '7828d110117889667c3a5e5877f7be9bd0222c13'
LABELS = {'recent3_42': '4+4 Recent3', 'recent12_42': '4+4 Recent12',
          'recent20_42': '4+4 Recent20', 'random20_tail3_42': '4+4 RandomTail3',
          'random20_tail5_42': '4+4 RandomTail5', 'mender_recent3_42': '6+6 Recent3',
          'mender_recent20_42': '6+6 Recent20'}


def read(path): return json.loads(Path(path).read_text())
def write(path, data): Path(path).write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')


def bootstrap_mean(delta, draws=10000):
    """Exact empirical paired-user bootstrap via counts of identical credits.

    Multinomial counts avoid allocating draws x users. For four-model contrasts
    first form each user's contrast, preserving the covariance between models.
    """
    values, counts = np.unique(delta, return_counts=True)
    rng = np.random.default_rng(4242); boots = []
    for begin in range(0, draws, 100):
        sample = rng.multinomial(len(delta), counts/len(delta), size=min(100, draws-begin))
        boots.extend((sample @ values / len(delta)).tolist())
    return np.quantile(boots, [.025, .975]).tolist()


def contrast(a, b):
    d = a-b
    return dict(a=float(a.mean()), b=float(b.mean()), delta=float(d.mean()),
        relative_delta=float(a.mean()/b.mean()-1), conditional_user_ci95=bootstrap_mean(d),
        improved=int((d > 0).sum()), worsened=int((d < 0).sum()), tied=int((d == 0).sum()))


def replay(path, split, expected, raw, sid, index, counters):
    with np.load(path, allow_pickle=False) as z:
        pred, scores = z['semantic_ids'], z['scores']
        np.testing.assert_array_equal(z['user_ids'], raw[split][1])
        np.testing.assert_array_equal(z['target_item_ids'], raw[split][2])
    assert pred.shape == (22363, 20, 4) and np.isfinite(scores).all()
    assert np.all(np.diff(scores, axis=1) <= 1e-6)
    actual = evaluate_ranked_sids(index, pred, raw[split][2], cutoffs=(5, 10))
    for key, value in actual.items(): assert abs(value-expected[key]) < 1e-12, (path, key)
    hit = (pred == sid[raw[split][2], None, :]).all(-1)
    rank = np.where(hit.any(1), hit.argmax(1)+1, 21)
    credit = np.where(rank <= 10, 1/np.log2(rank+1), 0.)
    counters['files'] += 1
    for field, key in [('candidates', 'prediction_count'), ('invalid', 'invalid_predictions'), ('duplicates', 'duplicate_predictions')]:
        counters[field] += actual[key]
    return credit, rank


def audit_trial(root, trial, row, c, raw, sid, index, counters):
    name, width = trial['id'], trial['recent']; path = root/'trials'/name
    assert read(path/'trial-result.json') == row
    assert row['raw_case_sha256'] == {s: case_digest(raw[s]) for s in ('train', 'validation')}
    assert row['case_targets_sha256'] == {s: case_digest(raw[s][1:]) for s in ('train', 'validation')}
    assert row['test_case_sha256'] == case_digest(raw['test'])
    assert row['initial_state_sha256'] == c['expected_initial_state_sha256']
    assert row['sid_file_sha256'] == c['expected_sid_file_sha256']
    assert row['pair_sid_file_sha256'] == file_sha256(root/'frozen-indexes/pairs-42.npy')
    expected_flops = {12:257705640} if c.get('recent12_control') else {3:123168070, 20:640482218}
    # Supported profiler counts differ by44FLOPs for the Recent3VALprobe
    # versus the TRAINprobe. Retain both measurements without asserting a cause.
    assert abs(row['forward']['supported_forward_flops_batch1']-expected_flops[width]) <= 256
    assert row['forward']['encoder_positions'] == 4*width+1
    assert row['forward']['parameters'] == (5108992 if c.get('recent12_control') else 7338496)
    curve = read(path/'training-curves.json'); selection = row['selection']
    stored = read(path/'training-selection.json')
    assert stored['criterion'] == 'validation_item_ndcg@10'
    assert all(stored[k] == value for k, value in selection.items())
    assert [r['epoch'] for r in curve] == list(range(1, selection['training_epochs']+1))
    steps = math.ceil(len(raw['train'][2])/512)
    def lr(step):
        if step < 2*steps: return .003*(step+1)/(2*steps)
        return .001+.001*(1+math.cos(math.pi*min(1., (step-2*steps+1)/(58*steps))))
    updates = 0; ema = reference = last = None; evaluated = []; credits = {}
    for r in curve:
        ep = r['epoch']; assert r['train_examples'] == 131413
        assert r['updates']+r['amp_skipped_updates'] == ep*steps
        assert abs(r['learning_rate']-lr(updates)) < 1e-14
        assert abs(r['next_learning_rate']-lr(r['updates'])) < 1e-14
        updates = r['updates']
        if 'validation' not in r: continue
        assert ep % 5 == 0
        loss = r['validation_loss']; assert loss['examples'] == 22363 and loss['precision'] == 'fp32'
        assert abs(sum(loss['ce_by_level'])-loss['total_ce']) < 1e-6
        credits[ep] = replay(path/f'validation-epoch-{ep:03d}.npz', 'validation', r['validation'], raw, sid, index, counters)
        score = r['validation']['ndcg@10']; ema = score if ema is None else .5*score+.5*ema
        if reference is None or ema > reference+.0005: reference = ema; last = ep
        assert (r['validation_ema_ndcg10'], r['ema_reference_ndcg10'], r['ema_last_improvement_epoch']) == (ema, reference, last)
        if ep >= 40 and ep < len(curve): assert ep-last < 20
        evaluated.append(r)
    best = max(evaluated, key=lambda r:r['validation']['ndcg@10'])
    assert best['epoch'] == selection['best_epoch'] and best['validation']['ndcg@10'] == selection['validation_ndcg10']
    assert selection['final_early_stopping_state'] == dict(ema=ema, reference=reference, last_improvement_epoch=last)
    assert selection['validation_checks'] == len(evaluated) == len(list(path.glob('validation-epoch-*.npz')))
    assert selection['optimizer_updates'] == updates
    assert selection['amp_skipped_updates'] == curve[-1]['amp_skipped_updates']
    assert abs(sum(r['train_seconds'] for r in curve)-selection['optimizer_training_seconds']) < 1e-6
    if selection['stopping_reason'] == 'early_stopping': assert len(curve) >= 40 and len(curve)-last >= 20
    else: assert selection['stopping_reason'] == 'epoch_limit' and len(curve) == 120
    assert read(path/'test-metrics.json') == row['test'] and row['test']['epoch'] == best['epoch']
    assert not row['test']['seen_items_filtered']
    selected = dict(validation=credits[best['epoch']],
        test=replay(path/'test-best-predictions.npz', 'test', row['test']['metrics'], raw, sid, index, counters))
    checkpoints = []
    for kind, epoch in [('best', best['epoch']), ('last', len(curve))]:
        file = path/f'generator-{kind}.pt'; ck = torch.load(file, map_location='cpu', weights_only=False)
        assert ck['epoch'] == epoch and all(torch.isfinite(v).all() for v in ck['model'].values())
        assert ck['config']['active_trial'] == trial and ck['config']['seed'] == 42
        assert ck['config']['generator'] == c['generator'] and ck['config']['training'] == c['training']
        if kind == 'last':
            for key in ('optimizer','scaler','scheduler','torch_rng','cuda_rng','python_rng','numpy_rng'): assert key in ck
            assert ck['updates'] == updates and ck['early_stopping_state'] == selection['final_early_stopping_state']
            assert ck['best_epoch'] == best['epoch'] and ck['sid_matrix_sha256'] == hashlib.sha256(sid.tobytes()).hexdigest()
            for split in ('train', 'validation'):
                h, u, t = raw[split]
                assert ck[split+'_case_sha256'] == array_digest((encode(h, sid, recent=width), u, t))
            for state in ck['optimizer']['state'].values():
                assert all(torch.isfinite(v).all() for v in state.values() if torch.is_tensor(v))
        checkpoints.append(dict(trial=name, kind=kind, epoch=epoch, sha256=file_sha256(file), finite=True, bytes=file.stat().st_size))
        del ck
    assert row['test']['checkpoint_sha256'] == checkpoints[0]['sha256']
    summary = dict(model=name, label=LABELS[name], validation_ndcg10=selection['validation_ndcg10'],
        test_ndcg10=row['test']['metrics']['ndcg@10'], test_recall10=row['test']['metrics']['recall@10'],
        best_epoch=best['epoch'], stop_epoch=len(curve), stopping_reason=selection['stopping_reason'],
        ce_min_epoch=min(evaluated, key=lambda r:r['validation_loss']['total_ce'])['epoch'],
        best_val_ce=min(r['validation_loss']['total_ce'] for r in evaluated), last_val_ce=evaluated[-1]['validation_loss']['total_ce'],
        warm_epoch_seconds=row['timing']['warm_epoch_training_median_seconds'],
        training_seconds=selection['optimizer_training_seconds'], validation_seconds=selection['validation_seconds'],
        amp_skips=selection['amp_skipped_updates'], forward_gflops=row['forward']['supported_forward_flops_batch1']/1e9,
        parameters=row['forward']['parameters'])
    return summary, curve, selected, checkpoints


def analyze(artifacts, out):
    artifacts, out = Path(artifacts), Path(out); out.mkdir(parents=True, exist_ok=True); torch.set_num_threads(1)
    old = artifacts/'beauty-pair/results'; old_config = read(old/'experiment.json'); validate(old_config)
    assert read(old/'run-manifest.json')['git_sha'] == OLD_SOURCE
    old_study = read(old/'study-summary.json'); old_analysis = read(old/'analysis.json')
    texts, users, _, counts = load_beauty(artifacts/'beauty-pair/input', old_config['data'])
    raw = {s:cases(old/'prepared-cases', s) for s in ('train', 'validation', 'test')}
    for split in raw:
        for actual, expected in zip(raw[split], build_examples(users, 20, split)): np.testing.assert_array_equal(actual, expected)
    sidpath = old/'frozen-indexes/items-42.npy'; sid = np.load(sidpath); index = SidIndex.from_full_sids(sid, width=256)
    counters = dict(files=0, candidates=0, invalid=0, duplicates=0)
    records = []; curves = {}; selected = {}; checkpoints = []; sessions = {}; array_hashes = {}
    for row in old_analysis['records']:
        name = row['model']; path = old/'trials'/name; original = old_study['results'][name]
        assert read(path/'trial-result.json') == original
        assert file_sha256(path/'generator-best.pt') == original['test']['checkpoint_sha256']
        selection = read(path/'training-selection.json'); assert row['best_epoch'] == selection['best_epoch']
        selected[name] = {split:replay(file, split, metrics, raw, sid, index, counters) for split, file, metrics in [
            ('validation', path/f"validation-epoch-{row['best_epoch']:03d}.npz", selection['selected_validation']),
            ('test', path/'test-best-predictions.npz', original['test']['metrics'])]}
        curve = read(path/'training-curves.json'); curves[name] = curve
        evaluated = [r for r in curve if 'validation' in r]
        records.append(dict(model=name, label=LABELS[name], validation_ndcg10=row['validation_ndcg10'],
            test_ndcg10=row['test_ndcg10'], test_recall10=row['test_recall10'], best_epoch=row['best_epoch'],
            stop_epoch=row['completed_epochs'], stopping_reason=row['stopping_reason'],
            ce_min_epoch=min(evaluated,key=lambda r:r['validation_loss']['total_ce'])['epoch'],
            best_val_ce=min(r['validation_loss']['total_ce'] for r in evaluated), last_val_ce=evaluated[-1]['validation_loss']['total_ce'],
            warm_epoch_seconds=row['warm_epoch_seconds'], training_seconds=row['optimizer_training_seconds'],
            validation_seconds=row['validation_seconds'], amp_skips=row['amp_skipped_updates'],
            forward_gflops=row['forward_gflops'], parameters=5108992))
    for stem, source in SOURCES.items():
        root = artifacts/f'beauty-{stem}-results'; c = read(root/'experiment.json'); validate(c)
        config_file = 'beauty_recent12_control.json' if stem == 'recent12' else 'beauty_mender_depth_2t4.json'
        assert c == read(Path('experiments')/config_file)
        manifest = read(root/'run-manifest.json'); suite = read(root/'suite-results.json'); study = read(root/'study-summary.json')
        metrics = read(root/'metrics.json'); assert manifest['git_sha'] == source and manifest['config'] == c
        assert (root/'summary.md').is_file() and not suite['failures'] and study['results'] == suite['results']
        assert metrics['models'] == study['new_generator_fits'] == len(c['trials'])
        assert all(not isinstance(v, float) or math.isfinite(v) for v in metrics.values())
        assert metrics['test_evaluated'] is True and c['training'] == old_config['training'] and c['quantizer'] == old_config['quantizer']
        array_hashes[stem] = {}
        for rel, oldrel in [('frozen-indexes/items-42.npy','frozen-indexes/items-42.npy'),
                ('frozen-indexes/item-centroids-42.npy','frozen-indexes/item-centroids-42.npy'),
                ('pair-cache/item-embeddings.npy','pair-cache/item-embeddings.npy')]:
            digest = file_sha256(root/rel); assert digest == file_sha256(old/oldrel); array_hashes[stem][rel] = digest
        assert file_sha256(root/'frozen-indexes/items-42.npy') == c['expected_sid_file_sha256']
        assert np.load(root/'frozen-indexes/pairs-42.npy').shape == (0,4) and np.load(root/'frozen-indexes/pair-keys.npy').shape == (0,)
        for split in raw:
            for actual, expected in zip(cases(root/'prepared-cases', split), raw[split]): np.testing.assert_array_equal(actual, expected)
        for trial in c['trials']:
            row = suite['results'][trial['id']]
            assert metrics[trial['id']+'_validation_ndcg10'] == row['selection']['validation_ndcg10']
            assert metrics[trial['id']+'_test_ndcg10'] == row['test']['metrics']['ndcg@10']
            summary, curve, credits, ck = audit_trial(root, trial, row, c, raw, sid, index, counters)
            records.append(summary); curves[trial['id']] = curve; selected[trial['id']] = credits; checkpoints.extend(ck)
        sessions[stem] = dict(source=source, seconds=manifest['duration_seconds'], models=len(c['trials']))
    groups = []; comparisons = {}; differences = {}; arrays = {}
    pairings = [('recent20_42','recent3_42'), ('mender_recent20_42','mender_recent3_42'),
        ('mender_recent20_42','recent20_42'), ('mender_recent3_42','recent3_42'),
        ('random20_tail3_42','recent12_42'), ('recent12_42','recent20_42'), ('recent12_42','recent3_42')]
    user_rows = []
    for split in ('validation', 'test'):
        lengths = (raw[split][0] >= 0).sum(1); arrays[split+'_lengths'] = lengths
        values = {k:v[split][0] for k,v in selected.items()}
        comparisons[split] = {a+'_vs_'+b:contrast(values[a],values[b]) for a,b in pairings}
        delta = (values['mender_recent20_42']-values['mender_recent3_42'])-(values['recent20_42']-values['recent3_42'])
        differences[split] = dict(absolute_history_gain_change=float(delta.mean()), conditional_user_ci95=bootstrap_mean(delta))
        masks = [('all',lengths>=0), ('<=3',lengths<=3), ('4-6',(lengths>=4)&(lengths<=6)),
            ('7-10',(lengths>=7)&(lengths<=10)), ('11-20',lengths>=11)]
        for label, mask in masks:
            if not mask.any(): continue
            groups.append(dict(split=split, group=label, users=int(mask.sum()),
                ndcg10={k:float(v[mask].mean()) for k,v in values.items()},
                old_history_gain=float((values['recent20_42']-values['recent3_42'])[mask].mean()),
                new_history_gain=float((values['mender_recent20_42']-values['mender_recent3_42'])[mask].mean())))
        for name, credit in values.items():
            arrays[name+'_'+split+'_credit'] = credit; arrays[name+'_'+split+'_rank'] = selected[name][split][1]
            for i, value in enumerate(credit): user_rows.append(dict(model=name,split=split,user_id=int(raw[split][1][i]),
                target_item_id=int(raw[split][2][i]),context_items=int(lengths[i]),ndcg10=float(value),rank=int(selected[name][split][1][i])))
    for file, rows in [('model-metrics.csv',records),('user-metrics.csv',user_rows)]:
        with (out/file).open('w',newline='') as f:
            w = csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    np.savez_compressed(out/'user-ranks.npz',**arrays)
    result = dict(sessions=sessions, records=records, comparisons=comparisons, history_gain_change=differences,
        groups=groups, replay=counters, checkpoints=checkpoints, identical_original_arrays=array_hashes,
        limits=['One generator/index seed per architecture; no seed robustness or pure capacity isolation (dropout also changes).',
            'Conditional paired-user intervals exclude seed/item clustering/checkpoint selection uncertainty/multiplicity.',
            'FLOPs are supported fixed padded teacher-forced forward; exclude beam/backward/text and do not imply serving latency.',
            'Raw GRID/P5 sequences lack timestamps for an independent boundary audit.'])
    write(out/'analysis.json',result); plot(out, records, curves, groups)
    return result


def plot(out, records, curves, groups):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = ['recent3_42','recent12_42','recent20_42','random20_tail3_42','mender_recent3_42','mender_recent20_42']
    fig, axes = plt.subplots(2,2,figsize=(13,8),constrained_layout=True)
    for name in names:
        checks = [r for r in curves[name] if 'validation' in r]
        axes[0,0].plot([r['epoch'] for r in checks],[r['validation']['ndcg@10'] for r in checks],label=LABELS[name])
        axes[0,1].plot([r['epoch'] for r in checks],[r['validation_loss']['total_ce'] for r in checks],label=LABELS[name])
    for ax, field in [(axes[1,0],'validation_ndcg10'),(axes[1,1],'test_ndcg10')]:
        for r in records:
            if r['model'] not in names: continue
            ax.scatter(r['forward_gflops'],r[field]);ax.annotate(r['label'],(r['forward_gflops'],r[field]),xytext=(4,4),textcoords='offset points',fontsize=8)
        ax.set_xlabel('Supported padded B1 forward GFLOPs');ax.set_ylabel(field);ax.margins(.2)
    axes[0,0].set_ylabel('VAL NDCG@10');axes[0,1].set_ylabel('VAL teacher-forced CE');axes[0,0].legend(fontsize=8)
    for ax in axes[0]:ax.set_xlabel('Epoch')
    for ax in axes.flat:ax.grid(alpha=.2)
    for ext in ('png','pdf'):fig.savefig(out/('learning-and-compute.'+ext),dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for row,split in enumerate(('validation','test')):
        selected = [r for r in groups if r['split']==split and r['group'] not in ('all','<=3')]
        for col,names in enumerate([['recent3_42','recent20_42','mender_recent3_42','mender_recent20_42'],
                                    ['recent12_42','recent20_42','random20_tail3_42','random20_tail5_42']]):
            ax=axes[row,col]
            for name in names:ax.plot([r['group'] for r in selected],[r['ndcg10'][name] for r in selected],marker='o',label=LABELS[name])
            ax.set_title(split.upper()+': '+('architecture' if col==0 else 'compression/truncation'))
            ax.set_xlabel('Actual visible raw context length');ax.set_ylabel('NDCG@10');ax.grid(alpha=.2);ax.legend(fontsize=8)
    for ext in ('png','pdf'):fig.savefig(out/('quality-by-length.'+ext),dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser();parser.add_argument('--artifacts',type=Path,default=Path('/workspace/artifacts'))
    parser.add_argument('--output',type=Path,default=Path('/workspace/artifacts/beauty-controls-analysis'));args=parser.parse_args()
    result=analyze(args.artifacts,args.output)
    print(json.dumps({k:result[k] for k in ('sessions','records','comparisons','history_gain_change','groups','replay')},indent=2))
