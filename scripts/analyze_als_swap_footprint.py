"""CPU-only audit of ALS SID reassignment exposure, geometry, popularity and quality."""
import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from rhpr.gr.data import load_beauty,build_examples,file_sha256
from rhpr.gr.sid_graph_swap import NeighborGraph
from rhpr.gr.sid_corruption import corruption_owner
from rhpr.gr.resources import verify_resources

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,default=Path('/workspace/artifacts/sid-als-footprint'))
p.add_argument('--plot',action='store_true');args=p.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
base=Path('/workspace/artifacts');inp=base/'sid-als-screen/input';repo=Path(__file__).resolve().parents[1]
spec=json.loads((repo/'experiments/amazon_beauty_sid_als_prepare_cpu.json').read_text())
current=json.loads((repo/'experiments/amazon_beauty_sid_corruption_2t4.json').read_text())
input_manifest=verify_resources(inp,current['sid_corruption']['semantic_graph']['manifest_sha256'])
verify_resources(base/'sid-context-screen-input',current['frozen_indices']['manifest_sha256'])
for key in ['embeddings','t5_rqkmeans']:
    assert file_sha256(Path(spec['sources'][key]['path']))==spec['sources'][key]['sha256']
original=np.load(inp/'t5_rqkmeans-original.npy');owner=np.load(inp/'t5_rqkmeans_uniform-owner.npy')
np.testing.assert_array_equal(original,np.load(spec['sources']['t5_rqkmeans']['path']))
paths=np.load(inp/'t5_rqkmeans_uniform-sids.npy');np.testing.assert_array_equal(paths,original[owner])
np.testing.assert_array_equal(np.sort(owner),np.arange(len(original)))
changed=owner!=np.arange(len(owner));selected=np.flatnonzero(changed);n=len(owner)
assert len(selected)==480 and n==12101
texts,users,source_ids,counts=load_beauty(Path('/workspace/scratch/gr-beauty-input/data'),current['data'])
assert input_manifest['data_sha256']==current['data']['sha256']
assert json.loads((inp/'catalogue.json').read_text())['original_item_ids']==source_ids
contexts,uids,targets=build_examples(users,20,'validation')
freq=np.bincount(np.concatenate([u.train[1:] for u in users]),minlength=n)
events=np.bincount(np.concatenate([u.train for u in users]),minlength=n);user_freq=np.zeros(n,dtype=int)
for u in users:user_freq[np.unique(u.train)]+=1
z=np.load(spec['sources']['embeddings']['path']).astype(np.float64)
z/=np.maximum(np.linalg.norm(z,axis=1,keepdims=True),1e-30)
lcp=lambda a,b:np.cumprod(a==b,axis=1).sum(1)
depth=lcp(original,paths);cosine=np.sum(z[selected]*z[owner[selected]],axis=1)

def summary(x):
    x=np.asarray(x,dtype=float)
    return dict(n=len(x),mean=float(x.mean()),q0_q10_q25_q50_q75_q90_q100=np.quantile(x,[0,.1,.25,.5,.75,.9,1]).tolist())

def exposure(own):
    mask=own!=np.arange(n);answer={}
    for split in ['train','validation']:
        ctx,_,tgt=build_examples(users,20,split);hit=(ctx>=0)&mask[np.maximum(ctx,0)]
        h=hit.any(1);t=mask[tgt]
        answer[split]=dict(examples=len(tgt),history=int(h.sum()),target=int(t.sum()),either=int((h|t).sum()),
            both=int((h&t).sum()),history_only=int((h&~t).sum()),target_only=int((t&~h).sum()),
            neither=int((~(h|t)).sum()),changed_input_occurrences=int(hit.sum()),input_occurrences=int((ctx>=0).sum()))
    answer['full_train_user_histories']=dict(users=len(users),affected=sum(bool(np.any(mask[np.asarray(u.train)])) for u in users))
    answer['changed_items']=int(mask.sum());answer['changed_train_target_mass']=float(freq[mask].sum()/freq.sum())
    return answer

footprint=exposure(owner)
root_changed=np.any((contexts>=0)&(depth[np.maximum(contexts,0)]==0),axis=1)
h=((contexts>=0)&changed[np.maximum(contexts,0)]).any(1);t=changed[targets]
graphs={}
for tag in ['semantic','als']:
    with np.load(inp/f'{tag}-graph.npz') as f:graphs[tag]=NeighborGraph(f['source'],f['target'],f['weight'],n)
geometry=dict(cosine=summary(cosine),cosine_distance=summary(1-cosine),normalized_euclidean_distance=summary(np.sqrt(np.maximum(2-2*cosine,0))))
rank=[]
for begin in range(0,len(selected),48):
    ids=selected[begin:begin+48];similarity=z[ids]@z.T
    threshold=similarity[np.arange(len(ids)),owner[ids]].copy();similarity[np.arange(len(ids)),ids]=-np.inf
    rank.extend(1+np.sum(similarity>threshold[:,None]+1e-12,axis=1))
rank=np.asarray(rank)
geometry['destination_neighbor_rank']=summary(rank)
geometry['destination_in_top20']=int((rank<=20).sum());geometry['destination_in_top100']=int((rank<=100).sum())
geometry['definition']='Cosine between item i and the original item occupying its newly assigned complete path,owner[i]. T5vectors never change;this is not a codebook-centroid distance.'

# Same exact480items/path inventory: isolates reassignment from moved-item choice,
# exposure and popularity. These100controls are index diagnostics only,not fitted models.
rng=np.random.default_rng(2755);random_records=[];random_cos=[]
for draw in range(100):
    while True:
        perm=rng.permutation(selected)
        if np.all(perm!=selected):break
    own=np.arange(n);own[selected]=perm;pp=original[own];cc=np.sum(z[selected]*z[perm],axis=1)
    dd=lcp(original[selected],pp[selected]);random_cos.append(cc)
    random_records.append(dict(draw=draw,mean_cosine=float(cc.mean()),median_cosine=float(np.median(cc)),
        lcp_counts=np.bincount(dd,minlength=4).tolist(),**{tag+'_score':g.score(pp) for tag,g in graphs.items()}))
random_cos=np.concatenate(random_cos)
a=rng.integers(n,size=20000);b=rng.integers(n,size=20000)
while np.any(a==b):
    same=a==b;b[same]=rng.integers(n,size=int(same.sum()))
global_cos=np.sum(z[a]*z[b],axis=1)
geometry['catalogue_random_pair_cosine']=summary(global_cos)
geometry['same_changed_set_random_cosine']=summary(random_cos)
graph_scores={tag:dict(original=g.score(original),als=g.score(paths),
    random_same_items_mean=float(np.mean([r[tag+'_score'] for r in random_records])),
    random_draw_2p5_97p5=np.quantile([r[tag+'_score'] for r in random_records],[.025,.975]).tolist()) for tag,g in graphs.items()}

bins=[(0,0),(1,4),(5,9),(10,19),(20,49),(50,10**9)]
popularity=dict(train_target_positions_moved=summary(freq[changed]),train_target_positions_unmoved=summary(freq[~changed]),
    train_users_moved=summary(user_freq[changed]),train_users_unmoved=summary(user_freq[~changed]),
    train_events_moved=summary(events[changed]),bins=[])
for lo,hi in bins:
    mask=(freq>=lo)&(freq<=hi)
    popularity['bins'].append(dict(label=f'{lo}-{hi}' if hi<10**9 else '50+',catalogue_items=int(mask.sum()),
        moved_items=int((mask&changed).sum()),moved_share_within_bin=float((mask&changed).sum()/mask.sum()),
        moved_train_target_positions=int(freq[mask&changed].sum())))

moves=np.load(base/'sid-als-screen/t5_rqkmeans/uniform/accepted-swaps.npy');assert moves.shape==(256,4)
replayed=np.arange(n);steps=[]
for position,a,b,delta in moves:
    a,b=int(a),int(b);prefix=int(lcp(original[replayed[[a]]],original[replayed[[b]]])[0])
    steps.append(dict(proposal=int(position),a=a,b=b,pre_swap_path_lcp=prefix,t5_cosine=float(z[a]@z[b]),objective_delta=float(delta)))
    replayed[[a,b]]=replayed[[b,a]]
np.testing.assert_array_equal(replayed,owner)
geometry['accepted_pair_cosine']=summary([s['t5_cosine'] for s in steps])

index_summaries={}
random5=corruption_owner(n,.05,2753)[0]
base_root_mass=np.bincount(original[:,0],weights=freq,minlength=256)/freq.sum()
for tag,own in [('uniform',owner),('priority',np.load(inp/'t5_rqkmeans_priority-owner.npy')),('priority_random',np.load(inp/'t5_rqkmeans_random-owner.npy')),
                ('context20',np.load(base/'sid-context-screen-input/context_20-owner.npy')),('context5',np.load(base/'sid-context-screen-input/context_5-owner.npy')),
                ('random5',random5)]:
    mask=own!=np.arange(n);idx=np.flatnonzero(mask)
    np.testing.assert_array_equal(np.sort(own),np.arange(n))
    index_summaries[tag]=dict(exposure=exposure(own),lcp_counts=np.bincount(lcp(original[idx],original[own[idx]]),minlength=4).tolist(),
        destination_cosine=summary(np.sum(z[idx]*z[own[idx]],axis=1)),
        moved_train_target_positions=summary(freq[idx]),moved_train_users=summary(user_freq[idx]),
        first_prefix_train_mass_TV=float(.5*np.abs(np.bincount(original[own,0],weights=freq,minlength=256)/freq.sum()-base_root_mass).sum()))

def ndcg(r):return np.where(r<=10,1/np.log2(r.astype(float)+1),0)
def intervals(delta,labels,seed):
    rng=np.random.default_rng(seed);user=[];cluster=[]
    _,g=np.unique(labels,return_inverse=True);num=np.bincount(g);sums=np.bincount(g,weights=delta)
    for _ in range(50):
        ix=rng.integers(len(delta),size=(40,len(delta)));user.extend(delta[ix].mean(1))
        ix=rng.integers(len(num),size=(40,len(num)));cluster.extend(sums[ix].sum(1)/num[ix].sum(1))
    return dict(user_ci95=np.quantile(user,[.025,.975]).tolist(),target_cluster_ci95=np.quantile(cluster,[.025,.975]).tolist())
groups=dict(all=np.ones(len(uids),bool),history_only=h&~t,target_only=t&~h,both=h&t,neither=~(h|t),either=h|t,
            history_any=h,target_any=t,first_token_target_changed=changed[targets]&(depth[targets]==0))
quality=[]
f=np.load(base/'sid-context-screen/analysis/user-metrics.npz')
np.testing.assert_array_equal(f['user_id'],uids);np.testing.assert_array_equal(f['target_item_id'],targets)
suite=json.loads((base/'sid-context-screen/suite-results.json').read_text())
for model in ['rqkmeans_original','item_als']:
    assert abs(float(ndcg(f[f'{model}_rank_e20']).mean())-suite['results'][model]['validation_at_budget']['ndcg@10'])<1e-12
long=pq.read_table(base/'sid-als-long/analysis/user-metrics.parquet',columns=['user_id','target_item_id','original_rank_e20','swap_rank_e20','original_rank_e60','swap_rank_e60']).to_pydict()
np.testing.assert_array_equal(long['user_id'],uids);np.testing.assert_array_equal(long['target_item_id'],targets)
comparisons=[('fresh20_ALS_vs_original',ndcg(f['rqkmeans_original_rank_e20']),ndcg(f['item_als_rank_e20']),groups),
    ('longrun20_ALS_vs_original',ndcg(np.array(long['original_rank_e20'])),ndcg(np.array(long['swap_rank_e20'])),groups),
    ('longrun60_ALS_vs_original',ndcg(np.array(long['original_rank_e60'])),ndcg(np.array(long['swap_rank_e60'])),groups)]
priority=np.load(inp/'t5_rqkmeans_priority-owner.npy');random=np.load(inp/'t5_rqkmeans_random-owner.npy')
np.testing.assert_array_equal(priority!=np.arange(n),random!=np.arange(n))
pm=priority!=np.arange(n);ph=((contexts>=0)&pm[np.maximum(contexts,0)]).any(1);pt=pm[targets]
root=base/'sid-context-robustness/completed/trials'
q=[]
for name in ['als10_t5_rqkmeans_random','als10_t5_rqkmeans_priority']:
    with np.load(root/name/'context-user-metrics.npz') as x:
        np.testing.assert_array_equal(x['user_ids'],uids);np.testing.assert_array_equal(x['target_item_ids'],targets)
        col=list(x['context_lengths']).index(20);score=ndcg(x['rank'][:,col]);np.testing.assert_array_equal(score,x['ndcg@10'][:,col]);q.append(score)
comparisons.append(('priority10_ALS_vs_same_itemset_random',q[0],q[1],dict(all=np.ones(len(uids),bool),either=ph|pt,neither=~(ph|pt))))
for j,(name,a,b,masks) in enumerate(comparisons):
    for label,mask in masks.items():
        row=dict(comparison=name,cohort=label,users=int(mask.sum()),control=float(a[mask].mean()),candidate=float(b[mask].mean()),
            delta=float((b-a)[mask].mean()),relative_percent=float(100*(b[mask].mean()/a[mask].mean()-1)))
        if label in ['all','either']:row.update(intervals((b-a)[mask],targets[mask],2756+j))
        quality.append(row)

items=dict(item_id=selected,source_item_id=[str(source_ids[i]) for i in selected],item_text=[texts[i] for i in selected],
    original_sid=original[selected].tolist(),assigned_sid=paths[selected].tolist(),assigned_path_original_item_id=owner[selected],
    original_path_lcp=depth[selected],t5_cosine=cosine,t5_cosine_distance=1-cosine,semantic_neighbor_rank=rank,
    train_target_positions=freq[selected],train_user_count=user_freq[selected],train_event_count=events[selected])
pq.write_table(pa.table(items),out/'moved-items.parquet',compression='zstd')
pq.write_table(pa.table({k:[s[k] for s in steps] for k in steps[0]}),out/'accepted-swaps.parquet',compression='zstd')
np.savez_compressed(out/'user-exposure.npz',user_id=uids,target_item_id=targets,history_affected=h,target_affected=t,
    first_token_history_affected=root_changed,full_history_changed_counts=np.array([changed[np.asarray(u.train)].sum() for u in users]))
report=dict(primary_mapping='t5_rqkmeans_uniform;480final moved items,256accepted transpositions',
    footprint=footprint,final_path_lcp_counts=np.bincount(depth[changed],minlength=4).tolist(),geometry=geometry,
    popularity=popularity,graph_scores=graph_scores,index_summaries=index_summaries,quality=quality,
    random_controls=dict(draws=100,seed=2755,controls=random_records,trained=False,
        interpretation='Same480selected items and original path inventory. Exact exposure/popularity matched;depth of movement not matched. Draw ranges are not quality confidence intervals.'),
    limitations=['Post-hoc and unadjusted,fixed fitted models;bootstrap does not capture neural/permutation seed variance.',
        'Unchanged input/target SIDrows can change through globally retrained shared weights.',
        'Fit ALS graph improvement is selected by construction;not proof of recommendation gain.',
        'Priority10vsrandom matches472items,not the main uniform480;cannot mix baselines or epochs.',
        'No new model fitting,inference,test access or GPU work.'],
    provenance=dict(embeddings_sha256=file_sha256(Path(spec['sources']['embeddings']['path'])),
        original_sids_sha256=file_sha256(inp/'t5_rqkmeans-original.npy'),owner_sha256=file_sha256(inp/'t5_rqkmeans_uniform-owner.npy'),
        frozen_resource_manifests_verified=True,data_hashes_and_catalogue_order_verified=True,
        accepted_moves_replayed_exactly=True,all_users_targets_aligned=True))
(out/'audit.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
lines=['# ALS swap footprint audit','',
    'Primary frozen index:t5_rqkmeans_uniform,480items/256accepted swaps. CPU-only,TRAINexposure and saved validation quality.','',
    'Detailed data:audit.json,moved-items.parquet,accepted-swaps.parquet,user-exposure.npz. '
    'T5 embeddings stay frozen;distance compares item i to the previous owner of its new complete SID. '
    'High absolute cosine must be read against same-domain random pairs.100same-itemset random indices are not trained generators.','',
    'Reproduce:OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 PYTHONPATH=src /workspace/rhpr-check/bin/python scripts/analyze_als_swap_footprint.py --plot']
(out/'README.md').write_text('\n'.join(lines)+'\n')
if args.plot:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(2,2,figsize=(11,8),layout='constrained')
    xx=np.arange(4);r=np.mean([a['lcp_counts'] for a in random_records],axis=0)
    ax[0,0].bar(xx-.18,np.bincount(depth[changed],minlength=4)/len(selected)*100,.36,label='ALS')
    ax[0,0].bar(xx+.18,r/len(selected)*100,.36,label='Random,same480items')
    ax[0,0].set_xticks(xx,['First','Second','Third','Suffix']);ax[0,0].set_title('Earliest changed token');ax[0,0].set_ylabel('Moved items (%)');ax[0,0].legend()
    for x,label in [(cosine,'ALS final assignment'),(random_cos,'Random,same480items'),(global_cos,'Random catalogue pair')]:
        x=np.sort(x);ax[0,1].plot(x,np.arange(1,len(x)+1)/len(x),label=label)
    ax[0,1].set_title('T5 pair similarity');ax[0,1].set_xlabel('Cosine');ax[0,1].set_ylabel('Cumulative fraction');ax[0,1].legend(fontsize=8)
    labels=[x['label'] for x in popularity['bins']];xx=np.arange(len(labels))
    ax[1,0].bar(xx-.18,[x['moved_items']/len(selected)*100 for x in popularity['bins']],.36,label='Moved480')
    ax[1,0].bar(xx+.18,[x['catalogue_items']/n*100 for x in popularity['bins']],.36,label='Whole catalogue')
    ax[1,0].set_xticks(xx,labels);ax[1,0].set_title('Popularity distribution');ax[1,0].set_xlabel('TRAIN next-item target positions');ax[1,0].set_ylabel('Items (%)');ax[1,0].legend()
    labels=['History','Target','Either'];xx=np.arange(3)
    for offset,tag,label in [(-.18,'uniform','ALS480'),(.18,'random5','Random5%(605items)')]:
        a=index_summaries[tag]['exposure']['validation'];ax[1,1].bar(xx+offset,[100*a[k]/a['examples'] for k in ['history','target','either']],.36,label=label)
    ax[1,1].set_xticks(xx,labels);ax[1,1].set_title('Validation rows affected at input/target');ax[1,1].set_ylabel('Validation examples (%)');ax[1,1].legend(fontsize=8)
    for a in ax.flat:a.grid(axis='y',alpha=.2)
    fig.suptitle('What the ALS swaps actually change')
    fig.savefig(out/'footprint.png',dpi=170);fig.savefig(out/'footprint.svg');plt.close(fig)
print(json.dumps({k:report[k] for k in ['footprint','final_path_lcp_counts','geometry','popularity','graph_scores','quality']},indent=2))
