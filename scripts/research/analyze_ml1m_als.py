"""CPU replay of clean ML1M MTL metrics, shared inputs and user uncertainty."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
from rhpr.gr.movie_data import case_digest
from rhpr.gr.sid import SidIndex

MODES=['sum','uw','gradnorm','pcgrad']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def replay(root,data,out):
    out.mkdir(parents=True,exist_ok=True);load=lambda p:json.loads(p.read_text())
    c=load(root/'experiment.json');manifest=load(root/'run-manifest.json');suite=load(root/'suite-results.json');plan=load(root/'plan.json')
    assert manifest['git_sha'] in {'20a4c08534cd6853e3fd64340f2dc429e6bed3a6','fc2a62bdd57af294caadd939bf1bbbda2de7032f','6ecbdaa0b54bd8d353270af203cf6d9d01646a24'} and manifest['config']==c==plan['config']
    from rhpr.gr.ml1m_clean_mtl import validate
    validate(c)
    assert not suite['failures'] and len(suite['results'])==len(c['trials'])
    for p in (root/'workers').glob('worker-*.json'):
        w=load(p);assert not w['failures'] and len(w['results'])==len(c['trials'])//2
    assert sha(data/'resources-manifest.json')==c['movie_input']['manifest_sha256']
    for r in load(data/'resources-manifest.json')['files']:
        p=data/r['path'];assert p.stat().st_size==r['size_bytes']and sha(p)==r['sha256']
    ids=load(data/'source-ids.json');assert ids==plan['source_ids']and len(ids)==2874
    cases={}
    for split,n in [('train',151166),('validation',2786),('test',2786)]:
        with np.load(data/(split+'-cases.npz'))as z:cases[split]=tuple(z[k]for k in ['contexts','user_ids','targets'])
        ex=cases[split];assert ex[0].shape==(n,20)and case_digest(ex)==c['data']['case_sha256'][split]
    np.testing.assert_array_equal(cases['validation'][1],cases['test'][1]);np.testing.assert_array_equal(cases['test'][0][:,:-1],cases['validation'][0][:,1:]);np.testing.assert_array_equal(cases['test'][0][:,-1],cases['validation'][2])
    feature_root=root/'features/shared-modalities';feature_hash=sha(feature_root/'features.npz');assert feature_hash==plan['features_sha256']
    assert load(feature_root/'source-ids.json')==ids
    with np.load(feature_root/'features.npz')as z:
        for k in ['text','poster','plot']:
            assert z[k].shape==(2874,768)and z[k+'_observed'].all()and np.isfinite(z[k]).all()
        assert z['people'].shape==(2874,128)and not z['people'].any()and not z['people_observed'].any()
    provenance=load(feature_root/'provenance.json');assert provenance['metadata_truncated_items']==0 and provenance['plot_encoder']['truncated_items']==0
    records=[];curves={};diagnostics=[];scores={};checks=[];ranks={};diag_curves={}
    for trial in c['trials']:
        mode=trial.get('cf_variant',trial['mtl_mode']);seed=trial['seed']
        name=trial['id'];p=root/'trials'/name;r=load(p/'trial-result.json');q=load(p/'quantizer.json');curve=load(p/'training-curves.json');curves[name]=curve;s=r['selection']
        assert r==suite['results'][name]and r['features_sha256']==feature_hash and r['case_sha256']==c['data']['case_sha256']
        assert s['training_epochs']==30 and s['optimizer_updates']==8880 and s['amp_skipped_updates']==0 and s['validation_checks']==6 and s['training_examples_per_epoch']==151166
        assert s['training_precision']=='fp16' and s['evaluation_precision']=='fp32'
        assert len(curve)==30 and all(x['updates']==296*x['epoch']and x['train_examples']==151166 for x in curve)
        assert [x['epoch']for x in curve if 'validation'in x]==[5,10,15,20,25,30]
        best=max((x for x in curve if 'validation'in x),key=lambda x:x['validation']['ndcg@10']);assert s['best_epoch']==best['epoch']and s['validation_ndcg10']==best['validation']['ndcg@10']
        assert q['steps']==3000 and q['precision']=='fp32' and q['warmup']['steps']==3500
        assert q['warmup']['codebooks_unchanged']and q['warmup']['fresh_rq_optimizer_verified']and q['warmup']['neural_transfer_verified']
        assert q['quantizer_initial_state_sha256']==c['quantizer']['expected_initial_state_sha256_by_seed'][str(seed)]
        assert load(p/'generator.json')['initial_state_sha256']==c['expected_initial_state_sha256_by_seed'][str(seed)]
        assert load(p/'generator.json')['config']['attn_implementation']=='sdpa'
        assert load(p/'sid-catalogue.json')['original_item_ids']==ids
        assert (load(p/'modality-normalization.json')['people'].get('inactive',False) if 'collaborative_input' not in c else load(p/'modality-normalization.json')['people']['observed_items']==2777)
        codes=np.load(p/'semantic-codes.npy');sid=np.load(p/'semantic-ids.npy');assert codes.shape==(2874,3)
        np.testing.assert_array_equal(SidIndex(codes,width=256).sids,sid)
        with np.load(p/'codebooks-final.npz')as z:np.testing.assert_array_equal(z['counts'],np.stack([np.bincount(codes[:,j],minlength=256)for j in range(3)]))
        lookup={tuple(v):j for j,v in enumerate(sid)}
        def score(path,split,expected):
            with np.load(path)as z:
                np.testing.assert_array_equal(z['user_ids'],cases[split][1]);np.testing.assert_array_equal(z['target_item_ids'],cases[split][2])
                pred=z['semantic_ids'];assert pred.shape==(2786,20,4)and np.isfinite(z['scores']).all()
                target=z['target_item_ids'];decoded=np.array([[lookup.get(tuple(v),-1)for v in row]for row in pred])
            hit=(pred==sid[target,None,:]).all(-1);rank=np.where(hit.any(1),hit.argmax(1)+1,0);values={}
            for k in [5,10]:
                ok=(rank>0)&(rank<=k);values[f'ndcg@{k}']=np.where(ok,1/np.log2(np.maximum(rank,1)+1),0.);values[f'recall@{k}']=ok.astype(float)
            for k,v in values.items():np.testing.assert_allclose(v.mean(),expected[k],atol=1e-12,rtol=0)
            invalid=int((decoded<0).sum());dup=sum(len(row[row>=0])-len(set(row[row>=0]))for row in decoded)
            assert invalid==expected['invalid_predictions']and dup==expected['duplicate_predictions']
            checks.append(dict(path=str(path.relative_to(root)),sha256=sha(path),invalid=invalid,duplicates=dup));return values['ndcg@10'],rank
        for x in curve:
            if 'validation'in x:
                values,rank=score(p/f"validation-epoch-{x['epoch']:03d}.npz",'validation',x['validation'])
                if x['epoch']==s['best_epoch']:scores[mode,seed,'validation']=values;ranks[mode,seed,'validation']=rank
        scores[mode,seed,'test'],ranks[mode,seed,'test']=score(p/'test-predictions.npz','test',r['test'])
        ev=load(p/'mtl-learning-curves.json');assert len(ev)==13;diag_curves[name]=ev
        cos=np.asarray([x['shared_encoder_gradient_cosines']for x in ev])[:,np.triu_indices(len(q['mtl_final_weights']),1)[0],np.triu_indices(len(q['mtl_final_weights']),1)[1]]
        dg=dict(mode=mode,seed=seed,final_weights=q['mtl_final_weights'],weight_sum=float(sum(q['mtl_final_weights'])),projection_totals=q['mtl_projection_totals'],sampled_last_layer_cosine_min=float(cos.min()),sampled_last_layer_negative_pairs=int((cos<0).sum()),sampled_pairs=int(cos.size),reconstruction_mse=q['head_reconstruction_mse'],used_codes=[len(np.unique(codes[:,j]))for j in range(3)],unique_triples=len(np.unique(codes,axis=0)),max_collision_group=q['semantic_code_statistics']['max_collision_group'])
        if mode=='pcgrad':
            dg.update(projected_at_3500=ev[6]['projected_updates'],projection_progress=[dict(step=x['step'],updates=x['projected_updates'],pairs=x['total_projections'])for x in ev])
        diagnostics.append(dg)
        records.append(dict(mode=mode,seed=seed,validation=s['validation_ndcg10'],test=r['test']['ndcg@10'],test_recall10=r['test']['recall@10'],best_epoch=s['best_epoch'],mean_epoch_seconds=s['optimizer_training_seconds']/30,tokenizer_seconds=q['warmup']['seconds']+q['fit_seconds'],invalid_test_predictions=r['test']['invalid_predictions']))
    for name,value in [('verification',checks),('training-curves',curves),('mtl-curves',diag_curves)]:
        (out/(name+'.json')).write_text(json.dumps(value,indent=2)+'\n')
    inv=[dict(path=str(p.relative_to(root)),size_bytes=p.stat().st_size,sha256=sha(p))for p in sorted(root.rglob('*'))if p.is_file()]
    (out/'download-manifest.json').write_text(json.dumps(inv,indent=2)+'\n')
    return dict(config=c,manifest=manifest,records=records,diagnostics=diagnostics,scores=scores,ranks=ranks,checks=checks,curves=curves,cases=cases,features_sha256=feature_hash)


def analyze(control,root,data,cf,out):
    out.mkdir(parents=True,exist_ok=True)
    old=replay(control,data,out/'reference');new=replay(root,data,out/'als')
    for key in ['training','generator','data','representation','fields','movie_input','execution','evaluation','plot_encoder','plot_assets','modern_runtime','quantizer']:
        assert old['config'][key]==new['config'][key],key
    assert old['features_sha256']==new['features_sha256']
    from rhpr.gr.resources import verify_resources
    verify_resources(cf,new['config']['collaborative_input']['manifest_sha256'])
    factors=np.load(cf/'als.npy');perm=np.load(cf/'permutation.npy');observed=np.load(cf/'observed.npy')
    np.testing.assert_array_equal(np.load(cf/'shuffled.npy'),factors[perm]);assert observed.sum()==2777 and not factors[~observed].any()and (perm[observed]!=np.flatnonzero(observed)).all()
    for variant in ['als','shuffled']:
        p=root/'trials'/f'{variant}_seed42';provenance=json.loads((p/'collaborative-provenance.json').read_text());assert provenance['factors_sha256']==sha(cf/(variant+'.npy'))and provenance['variant']==variant
        assert provenance['diagnostics']['train_case_sha256']==new['config']['data']['case_sha256']['train']and provenance['diagnostics']['no_val_test_read']
    records=[r for r in old['records']if r['mode']=='sum']+new['records'];scores={**old['scores'],**new['scores']};ranks={**old['ranks'],**new['ranks']}
    comparisons=[];rng=np.random.default_rng(20261105)
    for a,b in [('als','sum'),('shuffled','sum'),('als','shuffled')]:
        for split in ['validation','test']:
            delta=scores[a,42,split]-scores[b,42,split];boot=[]
            for _ in range(50):boot.extend(delta[rng.integers(0,2786,(100,2786))].mean(1).tolist())
            comparisons.append(dict(method=a,reference=b,split=split,delta=float(delta.mean()),relative_pct=float(100*delta.mean()/scores[b,42,split].mean()),ci95=np.quantile(boot,[.025,.975]).tolist(),better=int((delta>0).sum()),worse=int((delta<0).sum()),tied=int((delta==0).sum())))
    metrics=json.loads((root/'metrics.json').read_text());assert metrics['complete']==2
    for row in new['records']:
        assert metrics[row['mode']+'_seed42_test_ndcg10']==row['test']and metrics[row['mode']+'_seed42_validation_ndcg10']==row['validation']
    result=dict(records=records,comparisons=comparisons,diagnostics=[r for r in old['diagnostics']if r['mode']=='sum']+new['diagnostics'],features_sha256=new['features_sha256'],content_features_identical=True,run_minutes=new['manifest']['duration_seconds']/60,finished_at=new['manifest']['finished_at'],raw_files_checked=42,scope='One pipeline seed42; paired-user confidence intervals condition on fitted models/cohort and exclude training-seed and target-item dependence. Extra fourth unit reconstruction changes total scale3to4 in both newarms. Internal people slot contains ALS only.')
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    with(out/'metrics.csv').open('w')as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    users=[]
    for split in ['validation','test']:
        for i,u in enumerate(old['cases'][split][1]):
            row=dict(split=split,user_id=int(u),target_item_id=int(old['cases'][split][2][i]))
            for mode in ['sum','als','shuffled']:row[mode]=float(scores[mode,42,split][i]);row[mode+'_rank']=int(ranks[mode,42,split][i])
            users.append(row)
    with(out/'users.csv').open('w')as f:
        w=csv.DictWriter(f,fieldnames=list(users[0]));w.writeheader();w.writerows(users)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(1,2,figsize=(10,4));curves={**old['curves'],**new['curves']}
    for i,mode in enumerate(['sum','als','shuffled']):
        curve=[x for x in curves[mode+'_seed42']if 'validation'in x];ax[0].plot([x['epoch']for x in curve],[x['validation']['ndcg@10']for x in curve],marker='o',label=mode)
        r=next(r for r in records if r['mode']==mode);ax[1].bar(i,r['test'],color=f'C{i}');ax[1].text(i,r['test']+.0007,f"{r['test']:.4f}",ha='center')
    ax[0].set(xlabel='TIGER epoch',ylabel='Validation NDCG@10');ax[0].legend();ax[1].set(xticks=range(3),xticklabels=['Content','+ALS','+Shuffled ALS'],ylabel='Selected test NDCG@10',ylim=(0,max(r['test']for r in records)*1.2));fig.suptitle('ML1M ALS channel screen — seed42, 30epochs / val5');fig.tight_layout();fig.savefig(out/'als-comparison.png',dpi=180);fig.savefig(out/'als-comparison.pdf');plt.close(fig)
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ['control','root','data','cf','output']:p.add_argument(n,type=Path)
    a=p.parse_args();analyze(a.control,a.root,a.data,a.cf,a.output)
