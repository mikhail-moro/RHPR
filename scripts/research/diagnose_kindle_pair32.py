"""Saved-prediction error analysis, no training or new model inference."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def read(p): return json.loads(p.read_text())
def write(p,v): p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')


def pair_matrix(h,width,nitems):
    h=h[:,-width:];n=(h>=0).sum(1);out=np.full((len(h),width//4),-1,dtype=np.int64)
    for j in range(width//4):
        mask=2*j+1<n//2;rows=np.flatnonzero(mask);pos=width-n[mask]+2*j
        out[mask,j]=h[rows,pos].astype(np.int64)*nitems+h[rows,pos+1]
    return out


def lookup_counts(keys,counts,query):
    pos=np.searchsorted(keys,query);safe=np.minimum(pos,len(keys)-1)
    return np.where((pos<len(keys))&(keys[safe]==query),counts[safe],0)


def ranks(path,sids,users,targets):
    with np.load(path)as z:
        np.testing.assert_array_equal(z['user_ids'],users);np.testing.assert_array_equal(z['target_item_ids'],targets)
        pred=z['semantic_ids'];assert np.isfinite(z['scores']).all()
    hit=(pred==sids[targets,None,:]).all(-1)
    rank=np.where(hit.any(1),hit.argmax(1)+1,pred.shape[1]+1)
    credit=np.where(rank<=10,1/np.log2(rank+1),0.)
    return rank,credit,pred


def prediction_ids(pred,sids):
    weights=np.array([256**3,256**2,256,1],dtype=np.int64)
    codes=sids@weights;order=np.argsort(codes);sortedcodes=codes[order]
    pc=pred@weights;at=np.searchsorted(sortedcodes,pc);safe=np.minimum(at,len(codes)-1)
    valid=(at<len(codes))&(sortedcodes[safe]==pc)&(pred>=0).all(-1)&(pred<256).all(-1)
    return np.where(valid,order[safe],-1)


def error_summary(a,b,ar,br):
    delta=a-b;both=(ar<=10)&(br<=10);lost=(ar>10)&(br<=10);gained=(ar<=10)&(br>10)
    loss=-delta[delta<0].sum();gain=delta[delta>0].sum();order=np.argsort(delta)
    return dict(n=len(a),pair_ndcg=float(a.mean()),control_ndcg=float(b.mean()),absolute_delta=float(delta.mean()),
        pair_hits=int((ar<=10).sum()),control_hits=int((br<=10).sum()),lost_hits=int(lost.sum()),gained_hits=int(gained.sum()),shared_hits=int(both.sum()),
        improved_users=int((delta>0).sum()),worsened_users=int((delta<0).sum()),both_miss=int(((ar>10)&(br>10)).sum()),
        gross_loss_credit=float(loss),gross_gain_credit=float(gain),net_credit=float(delta.sum()),
        lost_hit_credit=float(-delta[lost].sum()),gained_hit_credit=float(delta[gained].sum()),
        shared_hit_rank_delta_credit=float(delta[both].sum()),top2_loss_credit=float(-delta[order[:2]].sum()),
        top5_loss_credit=float(-delta[order[:5]].sum()),top10_loss_credit=float(-delta[order[:10]].sum()),
        delta_without_two_largest_losses=float(delta[order[2:]].mean()),
        target_in_beam11_20_among_lost=int(((ar>10)&(ar<=20)&lost).sum()),
        target_outside_beam20_among_lost=int(((ar>20)&lost).sum()))


def analyze(root,data,item_embeddings,out,old_data=None):
    root,data,out=map(Path,(root,data,out));out.mkdir(parents=True,exist_ok=True)
    sids=np.load(root/'frozen-indexes/items-42.npy');nitems=len(sids);catalog=read(data/'catalog.json')
    audit=read(data/'user-audit.json');length={r['user_id']:r['events'] for r in audit}
    arrays={}
    for split in ('train','validation','test'):
        with np.load(data/f'{split}-cases.npz')as z:arrays[split]=tuple(z[k] for k in ('contexts','user_ids','targets'))
    train=arrays['train'];freq=np.bincount(train[2],minlength=nitems)
    matrix=pair_matrix(train[0],32,nitems);trainkeys,traincounts=np.unique(matrix[matrix>=0],return_counts=True)
    cachekeys=np.load(root/'frozen-indexes/pair-keys.npy');pairsids=np.load(root/'frozen-indexes/pairs-42.npy')
    itemvec=np.load(item_embeddings).astype(np.float32)
    assert itemvec.shape==(nitems,768)and np.isfinite(itemvec).all()
    itemvec/=np.maximum(np.linalg.norm(itemvec,axis=1,keepdims=True),1e-12)
    pairvec=np.load(root/'pair-cache/pair-embeddings.npy',mmap_mode='r')
    features={};ranked={};rows=[];summaries={};byepoch=[];val_deltas={};recommendations=[]
    with np.load(data/'histories.npz')as z:history_items=z['items'];offsets=z['offsets']
    for split in ('validation','test'):
        h,u,t=arrays[split];assert np.all(h[:,-32:]>=0)
        pkeys=pair_matrix(h,32,nitems);where=np.searchsorted(cachekeys,pkeys);np.testing.assert_array_equal(cachekeys[where],pkeys)
        pc=lookup_counts(trainkeys,traincounts,pkeys);pv=np.asarray(pairvec[where]).copy();pv/=np.maximum(np.linalg.norm(pv,axis=-1,keepdims=True),1e-12)
        old=itemvec[h[:,-32:-16]];recent=itemvec[h[:,-16:]];tv=itemvec[t]
        simold=np.einsum('nd,nkd->nk',tv,old);simrecent=np.einsum('nd,nkd->nk',tv,recent)
        simpair=np.einsum('nd,nkd->nk',tv,pv);first=np.einsum('nkd,nkd->nk',pv,old[:,::2]);second=np.einsum('nkd,nkd->nk',pv,old[:,1::2])
        within=np.einsum('nkd,nkd->nk',old[:,::2],old[:,1::2]);argold=simold.argmax(1)
        ownpair=simpair[np.arange(len(t)),argold//2]
        f=dict(history_events=np.array([length[int(x)]-(2 if split=='validation'else 1) for x in u]),
            target_frequency=freq[t],novel_pairs=(pc==0).sum(1),rare_pairs=(pc<=5).sum(1),mean_pair_exposure=pc.mean(1),
            pair_internal_similarity=within.mean(1),pair_component_retention=np.minimum(first,second).mean(1),
            pair_first_second_bias=(first-second).mean(1),pair_suffix_mean=pairsids[where,3].mean(1),
            mean_old_popularity=np.log1p(freq[h[:,-32:-16]]).mean(1),mean_recent_popularity=np.log1p(freq[h[:,-16:]]).mean(1),
            old_target_similarity=simold.max(1),recent_target_similarity=simrecent.max(1),
            target_semantic_dilution=simold.max(1)-ownpair,target_old_match_first=argold%2==0,
            target_closest_old=simold.max(1)>simrecent.max(1),target_in_history=(h==t[:,None]).any(1),
            nearest_old_position=argold.astype(int),nearest_recent_position=simrecent.argmax(1).astype(int))
        features[split]=f
        ranked[split]={}
        for model in ('recent20','recent32','pair32','pair64'):
            if split=='validation':path=root/('baselines'if model.startswith('recent')else'trials')/(model+'_wide_42')/'validation-epoch-055.npz'
            else:path=root/('control-tests'if model.startswith('recent')else'trials')/(model+'_wide_42')/'test-best-predictions.npz'
            ranked[split][model]=ranks(path,sids,u,t)
            ids=prediction_ids(ranked[split][model][2],sids)[:,:10];valid=ids>=0
            oldseen=(ids[:,:,None]==h[:,-32:-16][:,None,:]).any(-1)
            recentseen=(ids[:,:,None]==h[:,-16:][:,None,:]).any(-1)
            seen=[]
            for j,uid in enumerate(u):
                assert 1<=uid<len(offsets)
                past=history_items[offsets[uid-1]:offsets[uid]-(2 if split=='validation'else 1)]
                seen.append(np.isin(ids[j],past))
            recommendations.append(dict(split=split,model=model,mean_train_frequency=float(freq[ids[valid]].mean()),
                median_train_frequency=float(np.median(freq[ids[valid]])),fraction_frequency_gt20=float((freq[ids[valid]]>20).mean()),
                fraction_seen_in_observed_history=float(np.mean(seen)),fraction_seen_in_old16=float(oldseen.mean()),
                fraction_seen_in_recent16=float(recentseen.mean())))
        ar,a,_=ranked[split]['pair32'];br,b,_=ranked[split]['recent32'];delta=a-b
        summaries[split]=error_summary(a,b,ar,br)
        for j,uid in enumerate(u):
            meta=catalog[int(t[j])]
            rows.append(dict(split=split,user_id=int(uid),target_item=int(t[j]),title=meta['title'],
                categories=' > '.join(meta['categories']),pair_rank=int(ar[j]),control_rank=int(br[j]),
                pair_ndcg=float(a[j]),control_ndcg=float(b[j]),delta=float(delta[j]),
                **{k:float(v[j]) for k,v in f.items()}))
        if split=='validation':
            for epoch in range(5,61,5):
                args=[]
                for model in ('pair32','recent32'):
                    rr,gg,_=ranks(root/('trials'if model=='pair32'else'baselines')/(model+'_wide_42')/f'validation-epoch-{epoch:03d}.npz',sids,u,t)
                    args.append((rr,gg))
                byepoch.append(dict(epoch=epoch,**error_summary(args[0][1],args[1][1],args[0][0],args[1][0])))
                val_deltas[epoch]=args[0][1]-args[1][1]
    # Fixed meaningful buckets; feature medians are defined on VAL covariates,
    # never selected to maximize an outcome difference. All groups exploratory.
    f=features['validation'];masks={s:{} for s in features}
    for split,x in features.items():
        masks[split].update(target_frequency_0=x['target_frequency']==0,target_frequency_1_5=(x['target_frequency']>=1)&(x['target_frequency']<=5),
            target_frequency_6_20=(x['target_frequency']>=6)&(x['target_frequency']<=20),target_frequency_21plus=x['target_frequency']>20,
            history64_95=x['history_events']<96,history96_127=(x['history_events']>=96)&(x['history_events']<128),
            history128_255=(x['history_events']>=128)&(x['history_events']<256),history256plus=x['history_events']>=256,
            no_novel_pair=x['novel_pairs']==0,has_novel_pair=x['novel_pairs']>0,
            closest_target_in_old=x['target_closest_old'],closest_target_in_recent=~x['target_closest_old'],
            closest_old_is_first=x['target_old_match_first'],closest_old_is_second=~x['target_old_match_first'])
        for popular in (False,True):
            for rare in (False,True):
                masks[split][f'context_popular_{popular}_target_rare_{rare}']=((x['mean_recent_popularity']>np.median(f['mean_recent_popularity']))==popular)&((x['target_frequency']<=5)==rare)&(x['target_frequency']>0)
    thresholds={}
    for name in ('rare_pairs','mean_pair_exposure','pair_internal_similarity','pair_component_retention','pair_first_second_bias',
                 'pair_suffix_mean','mean_old_popularity','mean_recent_popularity','target_semantic_dilution','old_target_similarity'):
        thresholds[name]=median=float(np.median(f[name]))
        for split,x in features.items():
            masks[split][name+'_low']=x[name]<=median;masks[split][name+'_high']=x[name]>median
    groups=[]
    for split,ms in masks.items():
        ar,a,_=ranked[split]['pair32'];br,b,_=ranked[split]['recent32']
        for name,mask in ms.items():
            if not mask.any():continue
            groups.append(dict(split=split,group=name,**error_summary(a[mask],b[mask],ar[mask],br[mask])))
    rng=np.random.default_rng(4242);context_intervals=[]
    for split in ('validation','test'):
        delta=ranked[split]['pair32'][1]-ranked[split]['recent32'][1]
        for name in ('mean_recent_popularity_low','mean_recent_popularity_high'):
            a=delta[masks[split][name]]
            ci=np.quantile(a[rng.integers(0,len(a),size=(5000,len(a)))].mean(1),[.025,.975])
            context_intervals.append(dict(split=split,group=name,n=len(a),delta=float(a.mean()),conditional_ci95=ci.tolist()))
    dynamics=[]
    for name in ('mean_recent_popularity_low','mean_recent_popularity_high','target_frequency_1_5','target_frequency_21plus','target_semantic_dilution_high'):
        mask=masks['validation'][name]
        trajectory={str(ep):float(delta[mask].mean())for ep,delta in val_deltas.items()}
        late=[value for ep,value in trajectory.items()if int(ep)>=30]
        dynamics.append(dict(group=name,validation_epoch_deltas=trajectory,mean_epoch30_60_delta=float(np.mean(late)),
                             negative_late_checks=sum(x<0 for x in late),late_checks=len(late)))
    # Selected extremes are examples, not independent evidence. Include
    # nearest old item and its ordered partner to make semantic checks readable.
    examples=[]
    for split in ('validation','test'):
        h,u,t=arrays[split];ar,a,pa=ranked[split]['pair32'];br,b,pb=ranked[split]['recent32'];delta=a-b
        for kind,order in [('loss',np.argsort(delta)[:8]),('gain',np.argsort(-delta)[:5])]:
            for j in order:
                if (kind=='loss'and delta[j]>=0)or(kind=='gain'and delta[j]<=0):continue
                at=int(features[split]['nearest_old_position'][j]);start=at-at%2;items=h[j,-32:-16][start:start+2]
                recent_at=int(features[split]['nearest_recent_position'][j])
                def item(k):return dict(item_id=int(k),title=catalog[int(k)]['title'],categories=catalog[int(k)]['categories'],train_target_positions=int(freq[k]))
                def best_predictions(pred):
                    mapped={tuple(row):i for i,row in enumerate(sids)}
                    return [item(mapped[tuple(x)]) if tuple(x)in mapped else {'invalid_sid':x.tolist()} for x in pred[j,:3]]
                examples.append(dict(split=split,kind=kind,user_id=int(u[j]),target=item(t[j]),pair_rank=int(ar[j]),control_rank=int(br[j]),
                    delta=float(delta[j]),nearest_old_item_in_pair=at%2,nearest_old_pair=[item(k) for k in items],
                    nearest_recent_item=item(h[j,-16+recent_at]),target_sid=sids[t[j]].tolist(),
                    nearest_old_cosine=float(features[split]['old_target_similarity'][j]),nearest_recent_cosine=float(features[split]['recent_target_similarity'][j]),
                    target_pair_dilution=float(features[split]['target_semantic_dilution'][j]),novel_pairs=int(features[split]['novel_pairs'][j]),
                    pair_top3=best_predictions(pa),control_top3=best_predictions(pb)))
    results=dict(summaries=summaries,validation_by_epoch=byepoch,groups=groups,val_feature_medians=thresholds,
        group_validation_dynamics=dynamics,recommendation_distribution=recommendations,
        context_group_intervals=context_intervals,
        pair_semantic_summary={s:{k:dict(mean=float(np.mean(v)),median=float(np.median(v))) for k,v in x.items()
                                 if k in ('pair_first_second_bias','pair_component_retention','pair_internal_similarity','target_semantic_dilution')}
                               for s,x in features.items()},
        limits=['Exploratory analysis of one already examined seed and TEST; no causal attribution or independent replication.',
                'Feature medians fixed from VAL covariates, many descriptive comparisons; sparse positive outcomes and no multiplicity-adjusted tests.',
                'Target similarities use target information for diagnosis, unavailable for a serving router.',
                'Rank21 means absent from saved beam20, not proof of exclusion at an earlier decoding stage.'])
    if old_data is not None:
        comparison=[]
        for label,directory in [('old_dense',Path(old_data)),('long_core',data)]:
            with np.load(directory/'train-cases.npz')as z:tr=z['targets']
            with np.load(directory/'test-cases.npz')as z:te=z['targets']
            nf=len(read(directory/'catalog.json'));fr=np.bincount(tr,minlength=nf)[te]
            comparison.append(dict(cohort=label,items=nf,train_positions=len(tr),test_targets=len(te),
                cold_test_targets=int((fr==0).sum()),test_targets_with_1_to_5_train_positions=int(((fr>=1)&(fr<=5)).sum()),
                test_targets_with_at_most5_train_positions=int((fr<=5).sum()),median_test_target_train_positions=float(np.median(fr))))
        results['cohort_comparison']=comparison
    write(out/'diagnosis.json',results);write(out/'extreme-examples.json',examples)
    for name,values in [('user-features.csv',rows),('groups.csv',groups)]:
        with (out/name).open('w',newline='')as stream:
            writer=csv.DictWriter(stream,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
    return results


def plot(results,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.2),constrained_layout=True)
    rows=results['validation_by_epoch'];epochs=[r['epoch'] for r in rows]
    axes[0].plot(epochs,[r['control_ndcg'] for r in rows],label='Recent32',marker='.',color='#2463a8')
    axes[0].plot(epochs,[r['pair_ndcg'] for r in rows],label='Pair32',marker='.',color='#278455')
    axes[0].set_xlabel('Epoch');axes[0].set_ylabel('VAL NDCG@10');axes[0].legend();axes[0].grid(alpha=.2)
    labels=[]
    for j,r in enumerate(results['context_group_intervals']):
        lo,hi=r['conditional_ci95'];axes[1].errorbar(r['delta'],j,xerr=[[r['delta']-lo],[hi-r['delta']]],fmt='o',color='#2463a8' if r['split']=='validation' else '#c67b24',capsize=3)
        labels.append(f"{r['split']}: {'less' if r['group'].endswith('_low') else 'more'} popular recent items (n={r['n']})")
    axes[1].set_yticks(range(len(labels)),labels);axes[1].axvline(0,color='grey',lw=1);axes[1].invert_yaxis()
    axes[1].set_xlabel('Pair32 - Recent32 NDCG@10');axes[1].grid(axis='x',alpha=.2)
    fig.suptitle('Exploratory: one examined seed; user CIs exclude seed/selection/multiple-comparison uncertainty',fontsize=10)
    fig.savefig(out/'pair32-diagnosis.png',dpi=180);fig.savefig(out/'pair32-diagnosis.pdf');plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--data',type=Path,required=True)
    p.add_argument('--item-embeddings',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--old-data',type=Path);p.add_argument('--plot',action='store_true');a=p.parse_args()
    r=analyze(a.results,a.data,a.item_embeddings,a.output,a.old_data)
    if a.plot:plot(r,a.output)
    print(json.dumps(dict(summaries=r['summaries'],validation_by_epoch=r['validation_by_epoch'],groups=r['groups'],pair_semantic_summary=r['pair_semantic_summary']),indent=2))
