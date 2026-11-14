"""Independent saved-prediction replay of frozen prefix interventions."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from rhpr.gr.kindle_prefix_mask import inputs,mask_old,original_prediction,validate
from rhpr.gr.movie_data import case_digest
from rhpr.gr.sid import SidIndex,evaluate_ranked_sids

SOURCE='fdb04d44caa03b16f8a08482fffff4b8d4cac864'
def read(path):return json.loads(path.read_text())
def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def analyze(root,pair,control,data):
    root,data=map(Path,(root,data));parents=dict(pair=str(pair),control=str(control))
    c=read(root/'experiment.json');validate(c);manifest=read(root/'run-manifest.json')
    assert manifest['git_sha']==SOURCE and manifest['config']==c
    suite=read(root/'suite-results.json');study=read(root/'study-summary.json')
    assert not suite['failures'] and suite['results']==study['results']
    assert study['training_performed']is False and study['tokenizer_refitted']is False and study['encoder_positions_preserved']
    for name,spec in c['parents'].items():
        for file,record in spec['files'].items():
            p=Path(parents[name])/file
            assert p.stat().st_size==record['size_bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==record['sha256']
    for file,sha in c['movie_input']['sha256'].items():assert hashlib.sha256((data/file).read_bytes()).hexdigest()==sha
    idx=SidIndex.from_full_sids(np.load(Path(pair)/'frozen-indexes/items-42.npy'),width=256)
    with np.load(data/'train-cases.npz')as z:frequency=np.bincount(z['targets'],minlength=len(idx.sids))
    records=[];comparisons=[];recovery=[];user_arrays={};metrics_file=read(root/'metrics.json');count=0
    assert metrics_file['models']==2 and metrics_file['evaluations']==8 and metrics_file['training_performed']is False
    for split in ('validation','test'):
        ranks={};credits={}
        with np.load(data/f'{split}-cases.npz')as z:h,u,t=(z[k]for k in ('contexts','user_ids','targets'))
        meanpop=np.log1p(frequency[h[:,-16:]]).mean(1);low=meanpop<=c['evaluation']['recent_popularity_cutoff']
        groups=dict(all=np.ones(len(t),dtype=bool),less_popular_recent=low,more_popular_recent=~low)
        for model in ('pair32','recent32'):
            row=study['results'][model];assert row==read(root/'trials'/model/'trial-result.json')
            parent='pair'if model=='pair32'else'control';file=f'trials/{model}_wide_42/generator-best.pt'
            assert row['checkpoint_sha256']==c['parents'][parent]['files'][file]['sha256']
            assert row['selected_epoch']==55 and row['weights_unchanged'] and row['training_performed']is False
            original,_=inputs(c,model,parents,data,split)
            np.testing.assert_array_equal(original[1],u);np.testing.assert_array_equal(original[2],t)
            for variant in ('full','masked_old'):
                key=(model,variant);record=row['results'][split][variant]
                x=original[0] if variant=='full'else mask_old(original[0],model)
                assert record['case_sha256']==case_digest((x,u,t)) and record['original_case_sha256']==case_digest(original)
                assert record['encoder_positions']==x.shape[1]+1 and record['visible_sid_positions']==int((x[0]>=0).sum())
                with np.load(root/'trials'/model/f'{split}-{variant}.npz')as z:
                    pred=z['semantic_ids'];scores=z['scores'];np.testing.assert_array_equal(z['user_ids'],u);np.testing.assert_array_equal(z['target_item_ids'],t)
                assert np.isfinite(scores).all()
                if variant=='full':
                    with np.load(original_prediction(parents,model,split))as z:np.testing.assert_array_equal(pred,z['semantic_ids'])
                    assert record['replay']['all_ranked_sids_identical']
                metrics=evaluate_ranked_sids(idx,pred,t,cutoffs=(5,10))
                for k,value in metrics.items():assert abs(value-record['metrics'][k])<1e-12
                assert abs(metrics_file[f'{model}_{split}_{variant}_ndcg10']-metrics['ndcg@10'])<1e-12
                hit=(pred==idx.sids[t,None,:]).all(-1);rank=np.where(hit.any(1),hit.argmax(1)+1,21)
                credit=np.where(rank<=10,1/np.log2(rank+1),0.)
                assert abs(float(credit.mean())-metrics['ndcg@10'])<1e-12
                ranks[key]=rank;credits[key]=credit;count+=1
                for group,mask in groups.items():
                    records.append(dict(split=split,model=model,variant=variant,group=group,users=int(mask.sum()),
                        ndcg10=float(credit[mask].mean()),hits10=int((rank[mask]<=10).sum()),recall10=float((rank[mask]<=10).mean())))
                user_arrays[f'{split}_{model}_{variant}_rank']=rank
        rng=np.random.default_rng(4242)
        for group,mask in groups.items():
            n=int(mask.sum());boot=rng.integers(0,n,size=(5000,n))
            effects={model:credits[(model,'masked_old')]-credits[(model,'full')]for model in ('pair32','recent32')}
            effects['interaction']=effects['pair32']-effects['recent32']
            for model,delta in effects.items():
                a=delta[mask];comparisons.append(dict(split=split,group=group,model=model,users=n,mask_delta_ndcg10=float(a.mean()),
                    conditional_user_ci95=np.quantile(a[boot].mean(1),[.025,.975]).tolist(),improved_users=int((a>0).sum()),worsened_users=int((a<0).sum())))
        lost=(ranks[('recent32','full')]<=10)&(ranks[('pair32','full')]>10)
        extremes=(ranks[('recent32','full')]==1)&(ranks[('pair32','full')]>20)
        recovery.append(dict(split=split,original_lost_control_hits=int(lost.sum()),
            recovered_by_pair_mask=int((lost&(ranks[('pair32','masked_old')]<=10)).sum()),
            original_pair_hits_lost_after_mask=int(((ranks[('pair32','full')]<=10)&(ranks[('pair32','masked_old')]>10)).sum()),
            extreme_control_rank1_pair_outside20=int(extremes.sum()),
            extreme_masked_pair_ranks=ranks[('pair32','masked_old')][extremes].tolist()))
        user_arrays[split+'_user_ids']=u;user_arrays[split+'_targets']=t;user_arrays[split+'_less_popular_recent']=low
    np.savez_compressed(root/'user-ranks.npz',**user_arrays)
    result=dict(source=SOURCE,session_seconds=manifest['duration_seconds'],prediction_files_replayed=count,
        all_baselines_pins_cases_metrics_verified=True,training_performed=False,records=records,comparisons=comparisons,recovery=recovery,
        limits=['One previously inspected seed and TEST;conditional user CIs exclude seed/index/selection and multiple comparisons.',
                'Masking creates an inference distribution shift;this does not estimate performance of a retrained Recent16.',
                'Within-model positions and weights fixed;between-model weights and learned position geometry differ.'])
    write(root/'analysis.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--pair-parent',type=Path,required=True)
    p.add_argument('--control-parent',type=Path,required=True);p.add_argument('--data',type=Path,required=True);a=p.parse_args()
    r=analyze(a.results,a.pair_parent,a.control_parent,a.data);print(json.dumps(r,indent=2))
