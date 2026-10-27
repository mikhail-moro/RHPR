"""Verify completed identical-index seed fits and compare cached oracle headroom."""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from rhpr.gr.data import build_examples, file_sha256, load_beauty
from rhpr.gr.generator_suite import training_manifest
from rhpr.metrics import validate_metrics

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--completed', type=Path, default=Path('/tmp/rhpr-seed-oracle-completed'))
parser.add_argument('--output', type=Path, default=Path('/workspace/artifacts/sid-seed-oracle/analysis'))
parser.add_argument('--data', type=Path, default=Path('/workspace/scratch/gr-beauty-input/data'))
parser.add_argument('--assets', type=Path, default=Path('/workspace/artifacts/sid-context-screen-input'))
parser.add_argument('--historical', type=Path, default=Path('/workspace/artifacts/sid-context-screen/analysis/user-metrics.npz'))
args = parser.parse_args()
root, out = args.completed, args.output
out.mkdir(parents=True, exist_ok=True)
repo = Path(__file__).resolve().parents[1]
load = lambda p: json.loads(p.read_text())
write = lambda n, v: (out/(n+'.json')).write_text(json.dumps(v, indent=2, allow_nan=False)+'\n')
c = load(root/'run-manifest.json')['config']
assert load(root/'run-manifest.json')['git_sha'] == '178883fe87351ee81ea70830688782129370584d'
assert c == load(repo/'experiments/amazon_beauty_sid_seed_oracle_2t4.json')
old = load(repo/'experiments/amazon_beauty_sid_context_screen_2t4.json')
for key in ['seed','data','generator','training','quantizer','execution','frozen_indices','expected_train_cases_sha256']:
    assert c[key] == old[key], key
metrics = validate_metrics(load(root/'metrics.json'))
assert metrics['runs'] == 2 and metrics['parallel_workers'] == 2
suite = load(root/'suite-results.json')
models = ['original_seed42','original_seed43']
epochs = [5,10,15,20]
assert set(suite['results']) == set(models) and not suite['failures']
_, users, source_ids, counts = load_beauty(args.data, c['data'])
contexts, uids, targets = build_examples(users, 20, 'validation')
lengths = np.array([len(u.train) for u in users])
stream = training_manifest(users,c)
assert stream == load(root/'shared-training-stream.json')
assert stream['train_cases_sha256'] == c['expected_train_cases_sha256']
assert len(uids) == 22363 and stream['examples_per_epoch'] == 131413
initials = load(root/'initializations-by-trial.json')
assert initials['training_order_seed'] == 42
assert set(initials['initial_state_sha256_by_trial'].values()) == {t['expected_initial_state_sha256'] for t in c['trials']}
assert len(set(initials['initial_state_sha256_by_trial'].values())) == 2
assert file_sha256(args.assets/'resources-manifest.json') == c['frozen_indices']['manifest_sha256']
original = np.load(args.assets/'original.npy')
assert file_sha256(args.assets/'original.npy') == 'b196711dceedce81a991d995aa5c86ccd35288d5cdfe3f7aedb174de86470dae'
weights = 256 ** np.arange(3,-1,-1,dtype=np.int64)
truth = original[targets] @ weights
valid_paths = original @ weights
ranks, scores, verified, training = {}, {}, [], []


def ndcg(rank, k=10):
    return np.where(rank <= k, 1/np.log2(rank.astype(float)+1), 0)


for trial in c['trials']:
    name = trial['id']
    d = root/'trials'/name
    assert file_sha256(root/'indices'/name/'semantic-ids.npy') == file_sha256(args.assets/'original.npy')
    np.testing.assert_array_equal(np.load(root/'indices'/name/'owner.npy'), np.arange(len(source_ids)))
    result = suite['results'][name]
    assert result['seed'] == result['generator_seed'] == trial['generator_seed']
    assert result['training_order_seed'] == 42 and result['parameters'] == 4977920
    assert result['initial_state_sha256'] == trial['expected_initial_state_sha256']
    assert not result['test'] and result['test_evaluations'] == 0 and not result['quantizer_retrained']
    selection = result['selection']
    assert selection['epochs'] == 20 and selection['model_examples_consumed'] == 2628260
    assert selection['optimizer_updates'] == 10280 and selection['amp_skipped_updates'] == 0
    assert selection['validation_checks'] == 4 and selection['stop_reason'] == 'epoch_limit'
    curve = load(d/'training-curves.json')
    assert [r['epoch'] for r in curve if 'validation' in r] == epochs
    training.append(dict(model=name, neural_seed=trial['generator_seed'], data_order_seed=42,
                         initial_sha256=result['initial_state_sha256'], **selection))
    for epoch in epochs:
        file = d/f'validation-epoch-{epoch:03d}.npz'
        with np.load(file) as f:
            np.testing.assert_array_equal(f['user_ids'], uids)
            np.testing.assert_array_equal(f['target_item_ids'], targets)
            prediction = f['semantic_ids']
            assert prediction.shape == (len(uids),20,4)
            assert np.all((prediction>=0)&(prediction<256)) and np.isfinite(f['scores']).all()
            packed = prediction @ weights
            hit = packed == truth[:,None]
            rank = np.where(hit.any(1),hit.argmax(1)+1,999).astype(np.int16)
            valid = np.isin(packed,valid_paths)
            sorted_paths = np.sort(np.where(valid,packed,-1),axis=1)
            duplicates = (sorted_paths[:,1:]==sorted_paths[:,:-1]) & (sorted_paths[:,1:]>=0)
        logged = curve[epoch-1]['validation']
        assert int((~valid).sum()) == logged['invalid_predictions']
        assert int(duplicates.sum()) == logged['duplicate_predictions']
        for k in [5,10]:
            assert abs(float(ndcg(rank,k).mean())-logged[f'ndcg@{k}']) < 1e-12
            assert abs(float((rank<=k).mean())-logged[f'recall@{k}']) < 1e-12
        ranks[name,epoch], scores[name,epoch] = rank, ndcg(rank)
        verified.append(dict(file=file.relative_to(root).as_posix(),sha256=file_sha256(file),all_metrics_verified=True))

cached = np.load(root/'oracle-user-ranks.npz')
np.testing.assert_array_equal(cached['user_ids'],uids)
np.testing.assert_array_equal(cached['target_item_ids'],targets)
for (model,epoch),rank in ranks.items():
    np.testing.assert_array_equal(cached[f'{model}_rank_e{epoch}'],rank)
oracle = load(root/'paired-oracle.json')
rows, cohorts = [], []
masks = [('all',lengths>=0),('<10',lengths<10),('10+',lengths>=10),('3',lengths==3),('4',lengths==4),
         ('5-9',(lengths>=5)&(lengths<=9)),('10-19',(lengths>=10)&(lengths<=19)),('20+',lengths>=20)]
historical = np.load(args.historical)
for key,expected in [('user_id',uids),('target_item_id',targets),('train_history_length',lengths)]:
    np.testing.assert_array_equal(historical[key],expected)

for epoch in epochs:
    a,b = [scores[m,epoch] for m in models]
    ar,br = [ranks[m,epoch] for m in models]
    best = max(float(a.mean()),float(b.mean()))
    ideal = np.maximum(a,b)
    r = dict(epoch=epoch,a_ndcg10=float(a.mean()),b_ndcg10=float(b.mean()),oracle_ndcg10=float(ideal.mean()),
             absolute_gain=float(ideal.mean()-best),relative_gain_percent=float(100*(ideal.mean()/best-1)),
             a_hits=int((ar<=10).sum()),b_hits=int((br<=10).sum()),
             a_only_hits=int(((ar<=10)&(br>10)).sum()),b_only_hits=int(((ar>10)&(br<=10)).sum()),
             both_hits=int(((ar<=10)&(br<=10)).sum()),both_miss=int(((ar>10)&(br>10)).sum()),
             oracle_recall10=float(np.mean((ar<=10)|(br<=10))))
    saved = next(x for x in oracle['epochs'] if x['epoch']==epoch)
    for key,value in r.items():assert abs(value-saved[key]) < 1e-12,(key,epoch)
    ha,hb = [ndcg(historical[f'{m}_rank_e{epoch}']) for m in ['rqkmeans_original','item_als']]
    hbest = max(float(ha.mean()),float(hb.mean()))
    r['historical_original']=float(ha.mean());r['historical_als']=float(hb.mean())
    r['historical_oracle']=float(np.maximum(ha,hb).mean())
    r['historical_absolute_gain']=r['historical_oracle']-hbest
    r['historical_relative_gain_percent']=100*(r['historical_oracle']/hbest-1)
    rows.append(r)
    for label,mask in masks:
        aa,bb,oo=[float(x[mask].mean()) for x in [a,b,ideal]]
        cohorts.append(dict(epoch=epoch,cohort=label,users=int(mask.sum()),a=aa,b=bb,oracle=oo,
                            headroom_vs_best_cohort_pct=100*(oo/max(aa,bb)-1)))

# Joint user/target-cluster resampling conditions on these already-fitted pairs.
# This is population sensitivity, not a test of learnable routing or seed variance.
a,b=[scores[m,20] for m in models]
ha,hb=[ndcg(historical[f'{m}_rank_e20']) for m in ['rqkmeans_original','item_als']]
gap=np.maximum(a,b)-(a if a.mean()>=b.mean() else b)
hgap=np.maximum(ha,hb)-(ha if ha.mean()>=hb.mean() else hb)
matrix=np.stack([gap,hgap,gap-hgap],axis=1)
rng=np.random.default_rng(2752)
_,groups=np.unique(targets,return_inverse=True)
n=np.bincount(groups)
sums=np.stack([np.bincount(groups,weights=matrix[:,j]) for j in range(3)],axis=1)
user_draws,target_draws=[],[]
for _ in range(50):
    ix=rng.integers(len(uids),size=(40,len(uids)));user_draws.append(matrix[ix].mean(1))
    ix=rng.integers(len(n),size=(40,len(n)));target_draws.append(sums[ix].sum(1)/n[ix].sum(1)[:,None])
uc=np.quantile(np.concatenate(user_draws),[.025,.975],axis=0)
tc=np.quantile(np.concatenate(target_draws),[.025,.975],axis=0)
intervals=[dict(comparison=name,delta=float(matrix[:,j].mean()),user_ci95=uc[:,j].tolist(),target_cluster_ci95=tc[:,j].tolist())
           for j,name in enumerate(['same_index_headroom','historical_different_index_headroom','headroom_difference_same_minus_different'])]
write('epochs',rows);write('cohorts',cohorts);write('intervals',intervals);write('training',training)
write('verification',dict(source_commit='178883fe87351ee81ea70830688782129370584d',exact_config=True,
     exact_original_sid_bytes_and_owners=True,distinct_pinned_initial_neural_hashes=True,
     shared_case_and_epoch_order_hashes=True,all8_raw_beams_metrics_and_cached_ranks_verified=True,
     oracle_verified=True,prediction_files=verified,historical_users_targets_lengths_and_protocol_match=True,
     test_evaluations=0,new_training=False,new_inference=False,bootstrap_repeats=2000,bootstrap_seed=2752))
np.savez_compressed(out/'user-metrics.npz',user_id=uids,target_item_id=targets,train_history_length=lengths,
                    **{f'{m}_rank_e{e}':r for (m,e),r in ranks.items()})
for file in ['run-manifest.json','metrics.json','suite-results.json','paired-oracle.json','download-manifest.json']:
    shutil.copyfile(root/file,out.parent/file)
last=rows[-1]
lines=['# Same SID index, independent neural seeds: completed validation control','',
       'Both20epoch fits completed,one perT4,seeds42/43;fixed original RQ-KMeans bytes and data/order seed42. '
       'Each4977920parameters,2628260examples,10280updates,zero AMPskips. '
       f"Execution body{metrics['elapsed_seconds']:.1f}seconds. All8raw beam files,oracle and saved ranks independently verified. No test/new inference.",'',
       '|Epoch|Seed42|Seed43|Oracle|Headroom|Historical original/ALS headroom|','|---|---:|---:|---:|---:|---:|']
for r in rows:
    lines.append(f"|{r['epoch']}|{r['a_ndcg10']:.6f}|{r['b_ndcg10']:.6f}|{r['oracle_ndcg10']:.6f}|+{r['relative_gain_percent']:.2f}%|+{r['historical_relative_gain_percent']:.2f}%|")
lines+=['',f"Endpoint exclusive hits:seed42only{last['a_only_hits']},seed43only{last['b_only_hits']},both{last['both_hits']},bothmiss{last['both_miss']}. "
        f"Oracle hits{last['a_only_hits']+last['b_only_hits']+last['both_hits']};Recall10{last['oracle_recall10']:.6f}. "
        'The oracle selects a complete list with the better target rank;it does not merge or rerank items.','',
        'Large oracle headroom occurs with no SID-map changes. Therefore the earlier +28%headroom alone is not evidence of '
        'index-specific complementarity or justification for a hierarchy router. No additional oracle benefit from '
        'different indices is established by these pairs. One pair per condition does not estimate an exact causal fraction '
        'or prove equivalence. Winner predictability from pre-target features remains untested.','',
        'The fresh seed42fit differs from historical seed42prediction metrics despite matching initial weights,data and order;'
        'training nondeterminism remains a source of diversity. Epochs share fitted trajectories and users,not independent repeats. '
        'A deployable ensemble/router quality gain has not been measured.','',
        'Artifacts:epochs.json,cohorts.json,intervals.json,training.json,verification.json,user-metrics.npz. '
        'Intervals condition on fixed weights and are unadjusted;no training-seed uncertainty. '
        'Raw46numerical/report files25.88MB cached/tmp/rhpr-seed-oracle-completed. '
        'Four model-state filenames confirmed on private Kaggle;checkpoint contents not downloaded/verified.','',
        'Reproduce:PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /workspace/rhpr-check/bin/python scripts/analyze_sid_seed_oracle.py']
(out/'README.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(dict(endpoint=last,intervals=intervals),indent=2))
