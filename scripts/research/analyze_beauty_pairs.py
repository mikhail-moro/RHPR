"""CPU-only audit and paired analysis of the four fixed-tail Beauty fits."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from analyze_kindle_alignment import compare
from rhpr.gr.beauty_pairs import POLICY, TRIALS, TailTokenization, cache_sets, encode, validate
from rhpr.gr.data import build_examples, file_sha256, load_beauty
from rhpr.gr.ml100k_pair_raw import cases
from rhpr.gr.movie_data import case_digest
from rhpr.gr.pair_unique4 import unique_pair_sids
from rhpr.gr.sid import SidIndex, evaluate_ranked_sids

SOURCE = '7828d110117889667c3a5e5877f7be9bd0222c13'


def read(path): return json.loads(Path(path).read_text())
def write(path, value): Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def analyze(root, data):
    root, data = Path(root), Path(data); c = read(root/'experiment.json'); validate(c)
    manifest = read(root/'run-manifest.json'); study = read(root/'study-summary.json'); suite = read(root/'suite-results.json')
    assert manifest['git_sha'] == SOURCE and manifest['config'] == c
    assert suite['results'] == study['results'] and not suite['failures']
    assert study['new_generator_fits'] == 4 and study['policy'] == POLICY and not study['exact_TIGER_reproduction']
    assert (root/'summary.md').is_file()
    metrics = read(root/'metrics.json'); assert all(not isinstance(x, float) or math.isfinite(x) for x in metrics.values())
    texts, users, _, counts = load_beauty(data, c['data'])
    raw = {s: cases(root/'prepared-cases', s) for s in ('train', 'validation', 'test')}
    for split, arrays in raw.items():
        for got, expected in zip(arrays, build_examples(users, 20, split)): np.testing.assert_array_equal(got, expected)
    idxroot = root/'frozen-indexes'; sid = np.load(idxroot/'items-42.npy'); keys = np.load(idxroot/'pair-keys.npy'); pairs = np.load(idxroot/'pairs-42.npy')
    idx = SidIndex.from_full_sids(sid, width=256); sets = cache_sets(root/'prepared-cases', len(sid))
    np.testing.assert_array_equal(keys, np.unique(np.concatenate(list(sets.values()))))
    assert {s:len(v) for s,v in sets.items()} == c['expected_pair_counts'] and len(keys) == 65080
    import torch
    from rhpr.gr.pair_sid import predict_rq
    torch.set_num_threads(1)
    assignment_audit = {}
    for label, expected_rows in [('item', 12101), ('pair', 65080)]:
        vec = np.load(root/'pair-cache'/f'{label}-embeddings.npy', mmap_mode='r')
        assert vec.shape == (expected_rows, 768) and vec.dtype == np.float32 and np.isfinite(vec).all()
        centers = np.load(idxroot/f'{label}-centroids-42.npy'); codes = predict_rq(vec, centers, 'cpu')
        stored = sid if label == 'item' else pairs
        changed = np.flatnonzero((codes != stored[:, :3]).any(1)); exceptions = []
        for at in changed:
            # Normalizing a numerically zero residual can amplify CPU/CUDA
            # rounding into an arbitrary later code. Retain original frozen SIDs.
            level = int(np.flatnonzero(codes[at] != stored[at, :3])[0])
            x = torch.tensor(np.array(vec[at], copy=True))
            for previous in range(level):
                x = torch.nn.functional.normalize(x, dim=0)-torch.tensor(centers[previous, stored[at, previous]])
            norm = float(x.norm()); assert norm <= 1e-6, (label, int(at), level, norm)
            exceptions.append(dict(row=int(at), first_different_level=level, residual_norm=norm))
        assignment_audit[label] = dict(rows=expected_rows, exact_cpu_prefix_rows=expected_rows-len(changed),
            near_zero_residual_cpu_cuda_exceptions=exceptions, stored_sids_preserved=True)
        # Check append-only uniqueness allocation on the ACTUAL exported prefixes,
        # not on the few platform-sensitive recomputed residual directions.
        if label == 'item': expected_sids = SidIndex(stored[:, :3], width=256).sids
        else: expected_sids = unique_pair_sids(keys, stored[:, :3], sets['train'], np.union1d(sets['train'], sets['validation']))
        np.testing.assert_array_equal(sid if label == 'item' else pairs, expected_sids)
    records = []; curves = {}; credits = {}; ranks = {}; checkpoint_reports = {}; first_epoch_hashes = {}
    replayed_predictions = prediction_paths = invalid = duplicates = count_epochs = 0
    def replay(path, split, expected):
        nonlocal replayed_predictions, prediction_paths, invalid, duplicates
        with np.load(path) as z:
            pred = z['semantic_ids']; scores = z['scores']
            np.testing.assert_array_equal(z['user_ids'], raw[split][1]); np.testing.assert_array_equal(z['target_item_ids'], raw[split][2])
        assert pred.shape == (22363, 20, 4) and np.isfinite(scores).all() and np.all(np.diff(scores, axis=1) <= 1e-6)
        actual = evaluate_ranked_sids(idx, pred, raw[split][2], cutoffs=(5, 10))
        for k,v in actual.items(): assert abs(v-expected[k]) < 1e-12, (path, k)
        hit = (pred == sid[raw[split][2], None, :]).all(-1); rank = np.where(hit.any(1), hit.argmax(1)+1, 21)
        replayed_predictions += 1; prediction_paths += actual['prediction_count']; invalid += actual['invalid_predictions']; duplicates += actual['duplicate_predictions']
        return rank, np.where(rank <= 10, 1/np.log2(rank+1), 0.)
    steps = math.ceil(counts['train_examples']/512); initial = set()
    for trial in TRIALS:
        name = trial['id']; folder = root/'trials'/name; row = read(folder/'trial-result.json'); selection = row['selection']
        curve = read(folder/'training-curves.json'); curves[name] = curve
        print(json.dumps(dict(audit_trial=name, epochs=len(curve))), flush=True)
        assert row['raw_case_sha256'] == {s:case_digest(raw[s]) for s in ('train','validation')}
        assert row['case_targets_sha256'] == {s:case_digest(raw[s][1:]) for s in ('train','validation')}
        assert row['test_case_sha256'] == case_digest(raw['test'])
        assert row['sid_file_sha256'] == file_sha256(idxroot/'items-42.npy') and row['pair_sid_file_sha256'] == file_sha256(idxroot/'pairs-42.npy')
        initial.add(row['initial_state_sha256']); assert row['initial_state_sha256'] == c['expected_initial_state_sha256']
        assert row['forward']['parameters'] == 5108992
        assert row['forward']['encoder_positions'] == {'recent20_42':81,'recent3_42':13,'random20_tail3_42':49,'random20_tail5_42':53}[name]
        assert metrics[name+'_forward_flops'] == row['forward']['supported_forward_flops_batch1']
        assert [r['epoch'] for r in curve] == list(range(1,selection['training_epochs']+1))
        ema = reference = last_improvement = None; checks = []; prev = 0
        def lr(updates):
            if updates < 2*steps: return .003*(updates+1)/(2*steps)
            return .001+.002*.5*(1+math.cos(math.pi*min(1.,(updates-2*steps+1)/(58*steps))))
        views = TailTokenization(raw['train'][0],sid,keys,pairs,tail=trial['recent']) if trial['mode']=='random' else None
        for r in curve:
            ep = r['epoch']; assert r['train_examples'] == 131413
            assert r['updates']+r['amp_skipped_updates'] == ep*steps
            assert abs(r['learning_rate']-lr(prev)) < 1e-14
            assert abs(r['next_learning_rate']-lr(r['updates'])) < 1e-14
            assert min(abs(r['learning_rate_last_batch']-lr(k)) for k in (r['updates']-1,r['updates'])) < 1e-14
            prev = r['updates']
            if views is not None:
                order = np.random.default_rng(42+ep).permutation(len(raw['train'][2])); choices = views.schedule(ep)
                hist = np.zeros(views.capacity+1,dtype=np.int64); actual = 0
                for batch,begin in enumerate(range(0,len(order),512)):
                    part = order[begin:begin+512]; count = int(choices[batch]); hist[count] += len(part)
                    slots = np.r_[0,views.merge_slots(count,ep,batch).cumsum()]
                    actual += int(slots[np.maximum(views.lengths[part]-trial['recent'],0)//2].sum())
                log = r['training_views']; assert log['merge_count_examples'] == hist.tolist() and log['pair_summary_occurrences'] == actual
                assert log['schedule_sha256'] == hashlib.sha256(choices.tobytes()).hexdigest()
                if ep==1:
                    for batch,begin in enumerate(range(0,len(order),512)): views(ep,batch,order[begin:begin+512])
                    assert views.epoch_summary(ep)==log; first_epoch_hashes[name]=log['tokenized_epoch_sha256']
                count_epochs += 1
            if 'validation' not in r: continue
            score = r['validation']['ndcg@10']; ema = score if ema is None else .5*ema+.5*score
            if reference is None or ema>reference+.0005: reference=ema; last_improvement=ep
            assert (ema,reference,last_improvement)==(r['validation_ema_ndcg10'],r['ema_reference_ndcg10'],r['ema_last_improvement_epoch'])
            if ep>=40 and ep<len(curve): assert ep-last_improvement<20
            vrank,vcredit=replay(folder/f'validation-epoch-{ep:03d}.npz','validation',r['validation'])
            ranks[name,ep]=vrank; credits[name,ep]=vcredit; checks.append(r)
        best=max(checks,key=lambda r:r['validation']['ndcg@10']); assert best['epoch']==selection['best_epoch']
        assert selection['validation_ndcg10']==best['validation']['ndcg@10'] and metrics[name+'_validation_ndcg10']==selection['validation_ndcg10']
        assert selection['validation_checks']==len(checks)==len(list(folder.glob('validation-epoch-*.npz')))
        assert selection['final_early_stopping_state']==dict(ema=ema,reference=reference,last_improvement_epoch=last_improvement)
        assert selection['optimizer_updates']==curve[-1]['updates'] and selection['amp_skipped_updates']==curve[-1]['amp_skipped_updates']
        assert abs(selection['optimizer_training_seconds']-sum(r['train_seconds']for r in curve))<1e-6
        if selection['stopping_reason']=='early_stopping': assert len(curve)>=40 and len(curve)-last_improvement>=20
        else: assert selection['stopping_reason']=='epoch_limit' and len(curve)==120
        ranks[name,'validation']=ranks[name,best['epoch']]; credits[name,'validation']=credits[name,best['epoch']]
        test=row['test']; assert test['epoch']==best['epoch'] and not test['seen_items_filtered']
        assert metrics[name+'_test_ndcg10']==test['metrics']['ndcg@10']
        ranks[name,'test'],credits[name,'test']=replay(folder/'test-best-predictions.npz','test',test['metrics'])
        for kind,epoch in [('best',best['epoch']),('last',len(curve))]:
            path=folder/f'generator-{kind}.pt'; ck=torch.load(path,map_location='cpu',weights_only=False)
            assert ck['epoch']==epoch and all(torch.isfinite(v).all()for v in ck['model'].values())
            assert ck['config']['active_trial']==trial and ck['config']['active_training_view_policy']==POLICY
            if kind=='last':
                assert ck['updates']==selection['optimizer_updates'] and ck['amp_skipped_updates']==selection['amp_skipped_updates']
                for field in ('optimizer','scaler','scheduler','python_rng','numpy_rng','torch_rng','cuda_rng'): assert field in ck
                assert ck['early_stopping_state']==selection['final_early_stopping_state']
            checkpoint_reports[name+'/'+kind]=dict(epoch=epoch,size_bytes=path.stat().st_size,sha256=file_sha256(path));del ck
        assert test['checkpoint_sha256'] == checkpoint_reports[name+'/best']['sha256']
        records.append(dict(model=name,completed_epochs=len(curve),best_epoch=best['epoch'],stopping_reason=selection['stopping_reason'],
            validation_ndcg10=selection['validation_ndcg10'],test_ndcg10=test['metrics']['ndcg@10'],
            validation_hits10=int((ranks[name,'validation']<=10).sum()),test_hits10=int((ranks[name,'test']<=10).sum()),
            validation_recall10=best['validation']['recall@10'],test_recall10=test['metrics']['recall@10'],
            forward_gflops=row['forward']['supported_forward_flops_batch1']/1e9,encoder_positions=row['forward']['encoder_positions'],
            warm_epoch_seconds=row['timing']['warm_epoch_training_median_seconds'],first_epoch_seconds=row['timing']['first_epoch_training_seconds'],
            optimizer_training_seconds=selection['optimizer_training_seconds'],validation_seconds=selection['validation_seconds'],
            training_wall_seconds=selection['training_seconds'],peak_allocated_bytes=row['timing']['cuda_peak_allocated_bytes'],
            amp_skipped_updates=selection['amp_skipped_updates']))
    assert initial=={c['expected_initial_state_sha256']}
    comparisons={}; groups={}; tokenization={}; arrays={}
    for split in ('validation','test'):
        lengths=(raw[split][0]>=0).sum(1); arrays[split+'_lengths']=lengths; arrays[split+'_users']=raw[split][1]
        masks={'all':np.ones(len(lengths),dtype=bool),'length1_4':lengths<=4,'length5_6':(lengths>=5)&(lengths<=6),
            'length7_10':(lengths>=7)&(lengths<=10),'length11_20':lengths>=11,'tail3_eligible':lengths>=5,'tail5_eligible':lengths>=7}
        groups[split]={}
        for label,mask in masks.items():
            groups[split][label]=dict(users=int(mask.sum()),models={})
            for trial in TRIALS:
                name=trial['id']; groups[split][label]['models'][name]=dict(ndcg10=float(credits[name,split][mask].mean()),hits10=int((ranks[name,split][mask]<=10).sum()))
        for trial in TRIALS:
            name=trial['id']; arrays[name+'_'+split+'_rank']=ranks[name,split]; arrays[name+'_'+split+'_credit']=credits[name,split]
            if trial['mode']=='recent': continue
            for baseline in ('recent20_42','recent3_42'):
                comparisons[name+'_vs_'+baseline+'_'+split]=compare(credits[name,split],credits[baseline,split],ranks[name,split],ranks[baseline,split])
            for label in ('tail3_eligible','tail5_eligible','length11_20'):
                mask=masks[label]; groups[split][label]['models'][name]['vs_recent20']=compare(credits[name,split][mask],credits['recent20_42',split][mask],ranks[name,split][mask],ranks['recent20_42',split][mask])
            tail=trial['recent']; encoded=encode(raw[split][0],sid,keys,pairs,tail=tail)
            n_pairs=np.maximum(lengths-tail,0)//2
            assert np.array_equal((encoded>=0).sum(1)//4,lengths-n_pairs)
            tokenization[name+'_'+split]=dict(fixed_pair_occurrences=int(n_pairs.sum()),compressible_users=int((n_pairs>0).sum()),
                single_occurrences=int((lengths-2*n_pairs).sum()),mean_nonpadding_sid_tokens=float((lengths-n_pairs).mean()*4),padded_sid_width=encoded.shape[1])
    # Describe the persistent absolute VAL/TEST gap;this is not a causal split diagnosis.
    target_support=np.bincount(raw['train'][2],minlength=len(sid));split_targets={}
    for split in ('validation','test'):
        h,u,t=raw[split];support=target_support[t]
        split_targets[split]=dict(unique_targets=len(np.unique(t)),cold_target_users=int((support==0).sum()),
            median_target_train_positions=float(np.median(support)),mean_target_train_positions=float(np.mean(support)),
            target_already_in_visible_context_users=int((h==t[:,None]).any(1).sum()))
    result=dict(records=records,comparisons=comparisons,groups=groups,tokenization=tokenization,split_targets=split_targets,
        duration_seconds=manifest['duration_seconds'],prediction_files_replayed=replayed_predictions,prediction_paths_replayed=prediction_paths,
        invalid_paths=invalid,duplicate_paths=duplicates,actual_training_count_epochs_replayed=count_epochs,
        full_first_epoch_tensor_hashes=first_epoch_hashes,checkpoints=checkpoint_reports,
        assignment_audit=assignment_audit,pair_max_suffix=int(pairs[:,3].max()),
        limitations=['One pipeline seed;conditional paired user bootstrap excludes seed/item dependence;no multiplicity correction.',
            'Recent3is not compute-matched;no Recent12/13baseline;no proven noninferiority.',
            'FLOPs are padded teacher-forced forward,not measured serving latency;training has randomized weaker compression.',
            'Normalized GRID/P5records lack timestamps for independent raw-order verification.'])
    write(root/'analysis.json',result);np.savez_compressed(root/'user-ranks.npz',**arrays)
    with (root/'model-metrics.csv').open('w')as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    plot(root,records,curves)
    return result


def plot(root,records,curves):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for name,curve in curves.items():
        checks=[r for r in curve if 'validation'in r]
        axes[0,0].plot([r['epoch']for r in checks],[r['validation']['ndcg@10']for r in checks],label=name)
        axes[0,1].plot([r['epoch']for r in checks],[r['validation_loss']['total_ce']for r in checks],label=name)
    for ax,field in [(axes[1,0],'validation_ndcg10'),(axes[1,1],'test_ndcg10')]:
        for row in records:
            ax.scatter(row['forward_gflops'],row[field]);ax.annotate(row['model'],(row['forward_gflops'],row[field]),xytext=(4,4),textcoords='offset points',fontsize=8)
        ax.set_xlabel('B1 supported padded forward GFLOPs');ax.set_ylabel(field);ax.margins(.18)
    axes[0,0].set_ylabel('VAL NDCG@10');axes[0,1].set_ylabel('VAL teacher-forced CE');axes[0,0].legend(fontsize=8)
    for ax in axes[0]:ax.set_xlabel('Epoch')
    for ax in axes.flat:ax.grid(alpha=.2)
    fig.savefig(root/'beauty-curves.png',dpi=180);fig.savefig(root/'beauty-curves.pdf');plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--data',type=Path,required=True);a=p.parse_args()
    result=analyze(a.results,a.data);print(json.dumps({k:result[k]for k in ('records','comparisons','groups','split_targets')},indent=2))
