"""CPU replay of clean ML1M MTL metrics, shared inputs and user uncertainty."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
from rhpr.gr.movie_data import case_digest
from rhpr.gr.sid import SidIndex

MODES=['sum','uw','gradnorm','pcgrad']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def analyze(root,data,out):
    out.mkdir(parents=True,exist_ok=True);load=lambda p:json.loads(p.read_text())
    c=load(root/'experiment.json');manifest=load(root/'run-manifest.json');suite=load(root/'suite-results.json');plan=load(root/'plan.json')
    assert manifest['git_sha']=='20a4c08534cd6853e3fd64340f2dc429e6bed3a6' and manifest['config']==c==plan['config']
    assert not suite['failures'] and len(suite['results'])==4
    for p in (root/'workers').glob('worker-*.json'):
        w=load(p);assert not w['failures'] and len(w['results'])==2
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
    for mode in MODES:
        name=mode+'_seed42';p=root/'trials'/name;r=load(p/'trial-result.json');q=load(p/'quantizer.json');curve=load(p/'training-curves.json');curves[mode]=curve;s=r['selection']
        assert r==suite['results'][name]and r['features_sha256']==feature_hash and r['case_sha256']==c['data']['case_sha256']
        assert s['training_epochs']==30 and s['optimizer_updates']==8880 and s['amp_skipped_updates']==0 and s['validation_checks']==6 and s['training_examples_per_epoch']==151166
        assert s['training_precision']=='fp16' and s['evaluation_precision']=='fp32'
        assert len(curve)==30 and all(x['updates']==296*x['epoch']and x['train_examples']==151166 for x in curve)
        assert [x['epoch']for x in curve if 'validation'in x]==[5,10,15,20,25,30]
        best=max((x for x in curve if 'validation'in x),key=lambda x:x['validation']['ndcg@10']);assert s['best_epoch']==best['epoch']and s['validation_ndcg10']==best['validation']['ndcg@10']
        assert q['steps']==3000 and q['precision']=='fp32' and q['warmup']['steps']==3500
        assert q['warmup']['codebooks_unchanged']and q['warmup']['fresh_rq_optimizer_verified']and q['warmup']['neural_transfer_verified']
        assert q['quantizer_initial_state_sha256']==c['quantizer']['expected_initial_state_sha256_by_seed']['42']
        assert load(p/'generator.json')['initial_state_sha256']==c['expected_initial_state_sha256_by_seed']['42']
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
                if x['epoch']==s['best_epoch']:scores[mode,'validation']=values;ranks[mode,'validation']=rank
        scores[mode,'test'],ranks[mode,'test']=score(p/'test-predictions.npz','test',r['test'])
        ev=load(p/'mtl-learning-curves.json');assert len(ev)==13;diag_curves[mode]=ev
        cos=np.asarray([x['shared_encoder_gradient_cosines']for x in ev])[:,np.triu_indices(3,1)[0],np.triu_indices(3,1)[1]]
        dg=dict(mode=mode,final_weights=q['mtl_final_weights'],weight_sum=float(sum(q['mtl_final_weights'])),projection_totals=q['mtl_projection_totals'],sampled_last_layer_cosine_min=float(cos.min()),sampled_last_layer_negative_pairs=int((cos<0).sum()),sampled_pairs=int(cos.size),reconstruction_mse=q['head_reconstruction_mse'],used_codes=[len(np.unique(codes[:,j]))for j in range(3)],unique_triples=len(np.unique(codes,axis=0)),max_collision_group=q['semantic_code_statistics']['max_collision_group'])
        if mode=='pcgrad':
            dg.update(projected_at_3500=ev[6]['projected_updates'],projection_progress=[dict(step=x['step'],updates=x['projected_updates'],pairs=x['total_projections'])for x in ev])
        diagnostics.append(dg)
        records.append(dict(mode=mode,seed=42,validation=s['validation_ndcg10'],test=r['test']['ndcg@10'],test_recall10=r['test']['recall@10'],best_epoch=s['best_epoch'],mean_epoch_seconds=s['optimizer_training_seconds']/30,tokenizer_seconds=q['warmup']['seconds']+q['fit_seconds'],invalid_test_predictions=r['test']['invalid_predictions']))
    rng=np.random.default_rng(20261105);comparisons=[]
    for mode in MODES[1:]:
        for split in ['validation','test']:
            delta=scores[mode,split]-scores['sum',split];boot=[]
            for _ in range(50):boot.extend(delta[rng.integers(0,len(delta),size=(100,len(delta)))].mean(1).tolist())
            comparisons.append(dict(mode=mode,split=split,delta=float(delta.mean()),relative_pct=100*float(delta.mean()/scores['sum',split].mean()),ci95=np.quantile(boot,[.025,.975]).tolist(),better_users=int((delta>0).sum()),worse_users=int((delta<0).sum()),tied_users=int((delta==0).sum())))
    result=dict(source_code_sha=manifest['git_sha'],run_minutes=manifest['duration_seconds']/60,users=2786,items=2874,train_examples=151166,records=records,comparisons=comparisons,diagnostics=diagnostics,source_features_sha256=feature_hash,prediction_files_checked=len(checks),
        scope='One pipeline seed42. Paired-user bootstrap5000draws conditional on fitted models/cohort; excludes training-seed uncertainty, target-item clustering and multiplicity. Exploratory comparison, not a confirmed MTL advantage or proof clean filtering caused improvement.')
    for name,value in [('summary',result),('verification',checks),('training-curves',curves),('mtl-curves',diag_curves)]: (out/(name+'.json')).write_text(json.dumps(value,indent=2)+'\n')
    with(out/'metrics.csv').open('w')as f:w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    uu=[]
    for split in ['validation','test']:
        for j,uid in enumerate(cases[split][1]):
            row=dict(split=split,user_id=int(uid),target_item_id=int(cases[split][2][j]))
            for mode in MODES:row[mode]=float(scores[mode,split][j]);row[mode+'_rank']=int(ranks[mode,split][j])
            uu.append(row)
    with(out/'users.csv').open('w')as f:w=csv.DictWriter(f,fieldnames=list(uu[0]));w.writeheader();w.writerows(uu)
    inv=[dict(path=str(p.relative_to(root)),size_bytes=p.stat().st_size,sha256=sha(p))for p in sorted(root.rglob('*'))if p.is_file()];(out/'download-manifest.json').write_text(json.dumps(inv,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for i,mode in enumerate(MODES):
        cc=[x for x in curves[mode]if 'validation'in x];axes[0].plot([x['epoch']for x in cc],[x['validation']['ndcg@10']for x in cc],marker='o',label=mode)
        row=next(r for r in records if r['mode']==mode);axes[1].bar(i,row['test'],color=f'C{i}');axes[1].text(i,row['test']+.0007,f"{row['test']:.4f}",ha='center',fontsize=8)
        ev=diag_curves[mode];axes[2].plot([x['step']for x in ev],[np.mean(np.asarray(x['shared_encoder_gradient_cosines'])[np.triu_indices(3,1)])for x in ev],marker='.',label=mode)
    axes[0].set(xlabel='TIGER epoch',ylabel='Validation NDCG@10');axes[0].legend(fontsize=8)
    axes[1].set(ylabel='Selected test NDCG@10',xticks=range(4),xticklabels=MODES,ylim=(0,.075));axes[2].set(xlabel='AE + RQ optimizer calls',ylabel='Sampled last-layer gradient cosine');axes[2].legend(fontsize=8)
    fig.suptitle('ML1M clean contiguous holdouts — one seed, 30 epochs');fig.tight_layout();fig.savefig(out/'ml1m-clean-mtl.png',dpi=180);fig.savefig(out/'ml1m-clean-mtl.pdf');plt.close(fig)
    print(json.dumps(result,indent=2));return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('data',type=Path);p.add_argument('output',type=Path);a=p.parse_args();analyze(a.root,a.data,a.output)
