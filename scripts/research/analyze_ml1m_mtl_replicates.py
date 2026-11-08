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
    assert manifest['git_sha'] in {'20a4c08534cd6853e3fd64340f2dc429e6bed3a6','fc2a62bdd57af294caadd939bf1bbbda2de7032f'} and manifest['config']==c==plan['config']
    from rhpr.gr.ml1m_clean_mtl import validate
    validate(c)
    assert not suite['failures'] and len(suite['results'])==4*len(c['replicate_seeds'])
    for p in (root/'workers').glob('worker-*.json'):
        w=load(p);assert not w['failures'] and len(w['results'])==2*len(c['replicate_seeds'])
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
        mode=trial['mtl_mode'];seed=trial['seed']
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
        assert load(p/'modality-normalization.json')['people']['inactive']
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
        cos=np.asarray([x['shared_encoder_gradient_cosines']for x in ev])[:,np.triu_indices(3,1)[0],np.triu_indices(3,1)[1]]
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


def analyze(original,replicates,data,out):
    import itertools
    out.mkdir(parents=True,exist_ok=True)
    old=replay(original,data,out/'seed42');new=replay(replicates,data,out/'seeds101-202')
    for key in ['training','generator','data','representation','fields','movie_input','execution','evaluation','kaggle','plot_encoder','plot_assets','modern_runtime']:
        assert old['config'][key]==new['config'][key],key
    oq=dict(old['config']['quantizer']);nq=dict(new['config']['quantizer'])
    oq.pop('expected_initial_state_sha256_by_seed');nq.pop('expected_initial_state_sha256_by_seed');assert oq==nq
    # Compare contents as well as archive hashes: numerical features must be shared across runs.
    differences={}
    with np.load(original/'features/shared-modalities/features.npz')as a,np.load(replicates/'features/shared-modalities/features.npz')as b:
        assert set(a.files)==set(b.files)
        for k in a.files:
            np.testing.assert_array_equal(a[k],b[k]);differences[k]=0.0
    records=old['records']+new['records'];diagnostics=old['diagnostics']+new['diagnostics']
    scores={**old['scores'],**new['scores']};ranks={**old['ranks'],**new['ranks']};seeds=[42,101,202]
    aggregates=[];comparisons=[];rng=np.random.default_rng(20261105)
    for mode in MODES:
        rows=[next(r for r in records if r['mode']==mode and r['seed']==seed)for seed in seeds]
        aggregates.append(dict(mode=mode,validation_mean=float(np.mean([r['validation']for r in rows])),validation_sd=float(np.std([r['validation']for r in rows],ddof=1)),test_mean=float(np.mean([r['test']for r in rows])),test_sd=float(np.std([r['test']for r in rows],ddof=1)),mean_epoch_seconds=float(np.mean([r['mean_epoch_seconds']for r in rows]))))
        if mode=='sum':continue
        for split in ['validation','test']:
            deltas=np.stack([scores[mode,seed,split]-scores['sum',seed,split]for seed in seeds]);means=deltas.mean(1)
            # One shared user draw for all three seeds; do not count 3x2786 as independent users.
            fixed_seed_user_delta=deltas.mean(0);boot=[]
            for _ in range(50):boot.extend(fixed_seed_user_delta[rng.integers(0,2786,size=(100,2786))].mean(1).tolist())
            permutations=[float(np.mean(means*np.asarray(signs)))for signs in itertools.product([-1,1],repeat=3)]
            baseline=float(np.mean([scores['sum',seed,split].mean()for seed in seeds]))
            comparisons.append(dict(mode=mode,split=split,mean_delta=float(means.mean()),relative_pct=100*float(means.mean()/baseline),paired_seed_delta_sd=float(means.std(ddof=1)),conditional_user_ci95=np.quantile(boot,[.025,.975]).tolist(),seed_sign_flip_p_two_sided=sum(abs(x)>=abs(means.mean())-1e-14 for x in permutations)/8,per_seed=[dict(seed=seed,delta=float(means[i]),relative_pct=100*float(means[i]/scores['sum',seed,split].mean()))for i,seed in enumerate(seeds)],seeds_better=int((means>0).sum()),seeds_worse=int((means<0).sum())))
    result=dict(seeds=seeds,users_per_split=2786,items=2874,train_examples=151166,records=records,aggregates=aggregates,comparisons=comparisons,diagnostics=diagnostics,all_12_fits_complete=True,all_12_amp_skips_zero=True,prediction_files_checked=len(old['checks'])+len(new['checks']),run_minutes=new['manifest']['duration_seconds']/60,run_finished_at=new['manifest']['finished_at'],initial_run_minutes=old['manifest']['duration_seconds']/60,feature_arrays_identical=True,feature_sha256_by_run=[old['features_sha256'],new['features_sha256']],feature_max_abs_differences=differences,source_code_shas=[old['manifest']['git_sha'],new['manifest']['git_sha']],scope='Three pipeline seeds on one fixed cohort/split. Conditional user bootstrap shares user draws across seeds and excludes training-seed/target-item uncertainty. Exact paired-seed sign-flip test has only8 outcomes; descriptive small-n check, not definitive inference. No HPO or retraining.')
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    with(out/'metrics.csv').open('w')as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    users=[]
    for split in ['validation','test']:
        for i,uid in enumerate(old['cases'][split][1]):
            row=dict(split=split,user_id=int(uid),target_item_id=int(old['cases'][split][2][i]))
            for seed in seeds:
                for mode in MODES:row[f'{mode}_{seed}']=float(scores[mode,seed,split][i]);row[f'{mode}_{seed}_rank']=int(ranks[mode,seed,split][i])
            users.append(row)
    with(out/'users.csv').open('w')as f:
        w=csv.DictWriter(f,fieldnames=list(users[0]));w.writeheader();w.writerows(users)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(1,3,figsize=(15,4))
    for j,split in enumerate(['validation','test']):
        for i,mode in enumerate(MODES):
            vals=[scores[mode,seed,split].mean()for seed in seeds];ax[j].plot(seeds,vals,marker='o',label=mode,color=f'C{i}')
        ax[j].set(xticks=seeds,xlabel='Pipeline seed',ylabel=split.title()+' NDCG@10');ax[j].legend(fontsize=8)
    x=np.arange(3)
    for i,mode in enumerate(MODES[1:]):
        co=next(c for c in comparisons if c['mode']==mode and c['split']=='test')
        ax[2].bar(x+(i-1)*.23,[r['relative_pct']for r in co['per_seed']],width=.23,label=mode,color=f'C{i+1}')
    ax[2].axhline(0,color='black',linewidth=.8);ax[2].set(xticks=x,xticklabels=seeds,xlabel='Pipeline seed',ylabel='Test NDCG@10 change vs sum (%)');ax[2].legend(fontsize=8)
    fig.suptitle('Clean ML1M multimodal tokenization: seed replication, 30 epochs / val5');fig.tight_layout();fig.savefig(out/'three-seed-comparison.png',dpi=180);fig.savefig(out/'three-seed-comparison.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,4),sharey=True)
    curves={**old['curves'],**new['curves']}
    for ax,seed in zip(axes,seeds):
        for i,mode in enumerate(MODES):
            curve=[r for r in curves[f'{mode}_seed{seed}']if 'validation'in r]
            ax.plot([r['epoch']for r in curve],[r['validation']['ndcg@10']for r in curve],marker='.',label=mode,color=f'C{i}')
        ax.set(title=f'Seed{seed}',xlabel='TIGER epoch',ylabel='Validation NDCG@10');ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(out/'three-seed-training.png',dpi=180);fig.savefig(out/'three-seed-training.pdf');plt.close(fig)
    print(json.dumps({k:result[k]for k in ['aggregates','comparisons','run_minutes','prediction_files_checked','feature_arrays_identical']},indent=2))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('original',type=Path);p.add_argument('replicates',type=Path);p.add_argument('data',type=Path);p.add_argument('output',type=Path);a=p.parse_args();analyze(a.original,a.replicates,a.data,a.output)
