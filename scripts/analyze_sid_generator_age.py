"""Verify and summarize saved teacher-age proposals and likelihood searches; no inference."""
import argparse
import json
from pathlib import Path

import numpy as np

from rhpr.gr.data import build_examples,file_sha256,load_beauty
from rhpr.gr.generator_age import common_pool,budget_prefix
from rhpr.gr.generator_guided import incidence
from rhpr.gr.generator_suite import training_manifest

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--completed',type=Path,default=Path('/tmp/rhpr-generator-age-completed'))
p.add_argument('--output',type=Path,default=Path('/workspace/artifacts/sid-generator-age/analysis'))
p.add_argument('--data',type=Path,default=Path('/workspace/scratch/gr-beauty-input/data'))
p.add_argument('--plot',action='store_true');p.add_argument('--verify-checkpoints',action='store_true')
a=p.parse_args();r,out=a.completed,a.output;out.mkdir(parents=True,exist_ok=True)
load=lambda p:json.loads(p.read_text())
def write(name,v):(out/(name+'.json')).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
manifest=load(r/'run-manifest.json');c=manifest['config'];s=c['age_screen'];spec=s['search']
assert manifest['git_sha']=='316c1093a0a76ccf408562c5bce0a8640d72f322'
assert c==load(Path(__file__).resolve().parents[1]/'experiments/amazon_beauty_sid_generator_age_2t4.json')
_,users,ids,counts=load_beauty(a.data,c['data']);ex=build_examples(users,20,'train');val=build_examples(users,20,'validation')
assert training_manifest(users,c)==load(r/'shared-training-stream.json')
suite=load(r/'suite-results.json');assert not suite['failures'] and len(suite['results'])==3
provenance=load(r/'input-provenance.json');assert provenance['source_initial_state_sha256']==c['expected_initial_state_sha256']
training=load(r/'early-teacher/training-selection.json')
assert training['epochs']==5 and training['model_examples_consumed']==657065 and training['optimizer_updates']==2570 and training['amp_skipped_updates']==0
curve=load(r/'early-teacher/training-curves.json');assert len(curve)==5 and sum('validation' in v for v in curve)==1
original=np.load(r/'indices/epoch_001/semantic-ids.npy')
assert file_sha256(r/'indices/epoch_001/semantic-ids.npy')==s['sids']['sha256']
for e in [5,20]:np.testing.assert_array_equal(original,np.load(r/f'indices/epoch_{e:03d}/semantic-ids.npy'))
mined={};mining=[];frequency=np.bincount(ex[2],minlength=len(original))
expected_rows=np.sort(np.random.default_rng(spec['seed']).choice(len(ex[2]),spec['mining_rows'],replace=False))
for e in s['epochs']:
    d=r/'mining'/f'epoch-{e:03d}';pairs={}
    with np.load(d/'mining-probes.npz') as f:
        np.testing.assert_array_equal(f['train_row_ids'],expected_rows);np.testing.assert_array_equal(f['target_item_ids'],ex[2][expected_rows])
        invalid=int((f['proposed_item_ids']<0).sum())
        for i,j,g in zip(f['target_item_ids'],f['proposed_item_ids'],f['log_margin']):
            if j<0 or i==j or min(frequency[i],frequency[j])<spec['minimum_target_count'] or g<spec['confidence_log_margin']:continue
            k=tuple(sorted((int(i),int(j))));v=pairs.setdefault(k,dict(a=k[0],b=k[1],mistakes=0,priority=0.,maximum_log_margin=float(g)))
            v['mistakes']+=1;v['priority']+=min(float(g),spec['gap_clip']);v['maximum_log_margin']=max(v['maximum_log_margin'],float(g))
    ordered=sorted(pairs.values(),key=lambda x:(-x['priority'],-x['mistakes'],x['a'],x['b']))[:spec['max_proposals']]
    assert ordered==load(d/'candidate-pairs.json');mined[e]=ordered
    mining.append(dict(epoch=e,unique_pairs=len(pairs),invalid_greedy_paths=invalid,confident_positions=sum(v['mistakes'] for v in pairs.values())))
pool=common_pool(mined,s['pool_size'],s['pool_order_seed']);assert pool==load(r/'common-candidate-pool.json')
selected,cost=budget_prefix(pool,ex,len(original),spec['batch_size'],spec['max_scored_rows'])
assert selected==load(r/'shared-search-pairs.json') and len(selected)==115 and cost==1987754
assert cost==provenance['reserved_scored_rows_per_teacher']
assert file_sha256(r/'shared-search-pairs.json')==load(r/'workers/plan.json')['proposal_sha256']
incs=incidence(ex,len(original));catalog=load(a.data/'catalog.json');reports=[];traces={};accepted_union=set();item_reports=[]
for e in s['epochs']:
    name=f'epoch_{e:03d}';d=r/'searches'/name;report=load(d/'optimization.json');trace=[json.loads(x) for x in (d/'search-trace.jsonl').read_text().splitlines()]
    assert report=={k:v for k,v in suite['results'][name].items() if k not in ['epoch','checkpoint_sha256','proposal_sha256']}
    assert len(trace)==report['proposals']==len(selected) and report['scored_rows']==cost
    assert report['frozen_weights_and_rng_verified'] and report['cache_audit']['max_row_drift']==report['identity_drift']==0
    assert not report['validation_or_test_used_for_search']
    replay=np.arange(len(original));accepted=[];static_counts=[]
    for p,row in zip(selected,trace):
        i,j=row['a'],row['b'];assert (i,j)==(p['a'],p['b'])
        assert row['scored_rows']==p['scored_rows_per_evaluation'] and row['affected_rows']==len(np.union1d(incs[i],incs[j]))
        threshold=max(spec['min_total_gain'],spec['min_gain_per_affected']*row['affected_rows'])
        assert row['threshold']==threshold and row['accepted']==(row['delta_log_likelihood']>threshold)
        assert row['static_accepted']==(row['static_delta']>threshold)
        assert row['original_lcp']==int(np.cumprod(original[i]==original[j]).sum())
        if row['accepted']:replay[[i,j]]=replay[[j,i]];accepted.append(row);accepted_union.add((i,j))
    assert len(accepted)==report['accepted'] and sum(x['static_accepted'] for x in trace)==report['static_accepted']
    owner=np.load(d/'owner.npy');updated=np.load(d/'semantic-ids.npy');np.testing.assert_array_equal(owner,replay);np.testing.assert_array_equal(updated,original[owner])
    changed=np.flatnonzero(owner!=np.arange(len(owner)));assert len(changed)==report['changed_items']
    for depth in range(1,5):
        for x,y in zip(np.unique(original[:,:depth],axis=0,return_counts=True),np.unique(updated[:,:depth],axis=0,return_counts=True)):np.testing.assert_array_equal(x,y)
    with np.load(d/'train-likelihood-cache.npz') as f:
        np.testing.assert_array_equal(f['user_ids'],ex[1]);np.testing.assert_array_equal(f['target_item_ids'],ex[2])
        gain=f['final_log_probability']-f['initial_log_probability']
        assert abs(gain.sum()-report['total_train_log_likelihood_gain'])<1e-8
        assert abs(-f['initial_log_probability'].mean()-report['initial_mean_train_nll'])<1e-12
        assert abs(-f['final_log_probability'].mean()-report['final_mean_train_nll'])<1e-12
        affected=np.isin(ex[0],changed).any(1)|np.isin(ex[2],changed)
        assert affected.sum()==report['train_affected_rows'] and np.max(np.abs(gain[~affected]))==0
    assert abs(sum(x['delta_log_likelihood'] for x in accepted)-report['total_train_log_likelihood_gain'])<1e-8
    vhist=np.isin(val[0],changed).any(1);vtar=np.isin(val[2],changed)
    row=dict(epoch=e,**report,relative_nll_improvement_percent=100*(1-report['final_mean_train_nll']/report['initial_mean_train_nll']),
        val_changed_history_users=int(vhist.sum()),val_changed_target_users=int(vtar.sum()),val_changed_either_users=int((vhist|vtar).sum()),
        static_delta_quantiles=np.quantile([t['static_delta'] for t in trace],[0,.25,.5,.75,1]).tolist())
    reports.append(row);traces[e]=trace
    for i in changed:item_reports.append(dict(epoch=e,item=int(i),text=catalog[i]['text'],old_sid=original[i].tolist(),new_sid=updated[i].tolist(),train_targets=int(frequency[i])))
paired=[]
for i,j in sorted(accepted_union):
    values={str(e):next(t for t in traces[e] if (t['a'],t['b'])==(i,j)) for e in s['epochs']}
    paired.append(dict(a=i,b=j,source_text=[catalog[i]['text'],catalog[j]['text']],
        train_targets=[int(frequency[i]),int(frequency[j])],lcp=values['1']['original_lcp'],
        static_deltas={e:v['static_delta'] for e,v in values.items()},sequential_deltas={e:v['delta_log_likelihood'] for e,v in values.items()},
        static_accepted={e:v['static_accepted'] for e,v in values.items()}))
old_pairs=[(328,2126),(4943,5058)]
old_membership=[dict(pair=list(p),in_pool=any((x['a'],x['b'])==p for x in pool),in_evaluated_prefix=any((x['a'],x['b'])==p for x in selected)) for p in old_pairs]
checkpoint_records=[]
if a.verify_checkpoints:
    import torch
    from rhpr.gr.sid_study import state_digest
    for e in s['epochs']:
        path=r/'early-teacher'/f'generator-epoch-{e:03d}.pt' if e!=20 else Path('/tmp/rhpr-generator-guide-source/original-best20.pt')
        digest=file_sha256(path);assert digest==provenance['checkpoints'][str(e)]['sha256']
        ck=torch.load(path,map_location='cpu',weights_only=True);assert ck['epoch']==e and ck['updates']==e*514
        for key in ['seed','generator','data']:assert ck['config'][key]==c[key]
        state=state_digest(ck['model']);assert state==suite['results'][f'epoch_{e:03d}']['teacher_state_sha256']
        checkpoint_records.append(dict(epoch=e,bytes=path.stat().st_size,sha256=digest,model_state_sha256=state,updates=ck['updates']))
    write('checkpoint-verification',checkpoint_records)
summary=dict(teachers=reports,positive_pairs=paired,mining=mining,previous_accepted_pair_membership=old_membership,
    shared_pairs=len(selected),pool_size=len(pool),static_original_lcp_counts=np.bincount([t['original_lcp'] for t in traces[1]],minlength=4).tolist(),
    scored_rows_per_teacher=cost,elapsed_minutes=load(r/'metrics.json')['elapsed_seconds']/60,new_training_epochs=5,downstream_fits=0,test_evaluations=0)
write('summary',summary);write('items',item_reports);write('positive-pairs',paired)
write('verification',dict(exact_source_config=True,all_three_raw_mining_lists_recomputed=True,shared_pool_and_budget_prefix_recomputed=True,
    shared_training_cases_order_initial_hash=True,same_proposal_order_and_cost_all_ages=True,static_and_greedy_thresholds_verified=True,
    all_final_maps_replayed=True,all_catalogue_prefix_occupancies_preserved=True,all_train_cache_aggregates_and_zero_unaffected_deltas=True,
    runtime_frozen_weight_rng_and_full_audit_pass=True,checkpoint_contents_independently_downloaded=a.verify_checkpoints,
    all3_weight_hashes_and_age_updates_match_runtime=a.verify_checkpoints,new_inference=False))
np.savez_compressed(out/'paired-static-deltas.npz',a=[p['a'] for p in selected],b=[p['b'] for p in selected],
    original_lcp=[t['original_lcp'] for t in traces[1]],**{f'epoch_{e:03d}':np.array([t['static_delta'] for t in traces[e]]) for e in s['epochs']})
if a.plot:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for pair in paired:ax[0].plot(s['epochs'],list(pair['static_deltas'].values()),'o-',label=f"{pair['a']} / {pair['b']}")
    ax[0].axhline(0,color='gray',lw=.8);ax[0].set_title('Same original-map swap, different teachers');ax[0].set_ylabel('Total TRAIN log-likelihood change (nats)');ax[0].set_xlabel('Teacher epoch');ax[0].legend(fontsize=8)
    x=np.arange(3);ax[1].bar(x-.15,[v['static_accepted'] for v in reports],width=.3,label='Static');ax[1].bar(x+.15,[v['accepted'] for v in reports],width=.3,label='Sequential')
    ax[1].set_xticks(x,['1','5','20']);ax[1].set_xlabel('Teacher epoch');ax[1].set_ylabel('Accepted swaps');ax[1].set_title('Identical115proposals / row budget');ax[1].legend()
    fig.savefig(out/'overview.png',dpi=160);fig.savefig(out/'overview.svg');plt.close(fig)
print(json.dumps(summary,indent=2))
