"""Replay partial shared-codebook experiment; no training or generator inference."""
import argparse,json,csv
from pathlib import Path
import numpy as np
import torch
from scripts.research.analyze_mini_transfer import decode,load,sha
from rhpr.gr.shared_codebooks import assign,validate


def replay(root,data,parent,checkpoint,out):
    out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1)
    c=load(root/'experiment.json');validate(c);suite=load(root/'suite-results.json')
    assert list(suite['results'])==['codebook_init'] and len(suite['failures'])==1
    assert suite['failures'][0]['trial']=='live_codebook' and 'collision'in suite['failures'][0]['message']
    assert not (root/'metrics.json').exists() and not (root/'run-manifest.json').exists()
    workers=[load(root/'workers'/f'worker-{j}.json')for j in [0,1]]
    assert workers[0]['results']==suite['results'] and workers[1]['failures']==suite['failures']
    cases={}
    for split in ['validation','test']:
        with np.load(data/(split+'-cases.npz'),allow_pickle=False)as z:cases[split]=tuple(z[k]for k in ['contexts','user_ids','targets'])
    ids=np.load(root/'frozen-quantizer/initial-sids.npy');np.testing.assert_array_equal(ids,np.load(parent/'trials/sum_seed42/semantic-ids.npy'))
    rows=[];checks=[];users={};indexrows={}
    for name,base,epochs in [('control',parent/'trials/sum_seed42',30),('codebook_init',root/'trials/codebook_init',30),('live_codebook',root/'trials/live_codebook',29)]:
        curve=load(base/'training-curves.json');assert len(curve)==epochs
        assert all(r['updates']==r['epoch']*296 and r['train_examples']==151166 and r['amp_skipped_updates']==0 for r in curve)
        if name!='control':
            assert load(base/'generator.json')['initial_state_sha256']==c['initialization_pin']
            ir=load(base/'index-learning-curves.json');assert len(ir)==epochs;indexrows[name]=ir
        for r in curve:
            if 'validation'not in r:continue
            e=r['epoch'];sid=ids if name=='control' else np.load(base/f'semantic-ids-epoch-{e:03d}.npy')
            if name=='codebook_init':np.testing.assert_array_equal(sid,ids)
            v,ch=decode(base/f'validation-epoch-{e:03d}.npz',sid,cases['validation'],r['validation'])
            users[name,e]=v;checks.append(dict(arm=name,epoch=e,split='validation',**ch))
            rows.append(dict(arm=name,epoch=e,ndcg10=float(v.mean()),train_seconds=r['train_seconds'],**({'changed_items_previous':indexrows[name][e-1]['changed_items_previous']}if name!='control'else{})))
        if name!='live_codebook':
            selection=load(base/'training-selection.json');e=selection['best_epoch'];sid=ids if name=='control'else np.load(base/f'semantic-ids-epoch-{e:03d}.npy')
            if name=='codebook_init':
                np.testing.assert_array_equal(np.load(base/'semantic-ids.npy'),sid)
                expected=suite['results'][name]['test']
                assert selection['training_epochs']==30 and selection['optimizer_updates']==8880 and selection['validation_checks']==6
            else:expected=load(base/'trial-result.json')['test']
            v,ch=decode(base/'test-predictions.npz',sid,cases['test'],expected);users[name,'test']=v
            checks.append(dict(arm=name,epoch=e,split='test',**ch))
    cp=torch.load(checkpoint,map_location='cpu',weights_only=False);assert cp['epoch']==29 and cp['updates']==8584
    assert cp['config']==c
    np.testing.assert_array_equal(cp['model']['rhpr_active_sids'].numpy(),np.load(root/'trials/live_codebook/semantic-ids-epoch-029.npy'))
    books=cp['model']['item_sid_embedding_table_encoder.weight'][:768].reshape(3,256,128).float()
    latent=torch.tensor(np.load(root/'frozen-quantizer/frozen-latents.npy'))
    padded=torch.nn.functional.pad(latent,(0,64));failed=assign(padded,books);np.save(out/'attempted-epoch-030-semantic-codes.npy',failed)
    def stats(codes):
        _,counts=np.unique(codes,axis=0,return_counts=True)
        return dict(used_codes=[int(len(np.unique(codes[:,j])))for j in range(3)],max_collision_group=int(counts.max()),
            unique_semantic_ids=len(counts),overflow_groups=int((counts>256).sum()),items_in_overflow_groups=int(counts[counts>256].sum()))
    initial=torch.tensor(np.load(root/'frozen-quantizer/initial-books.npy'))
    lower=assign(latent,books[:,:,:64]);drift={}
    for j in range(3):
        drift[str(j+1)]=dict(initial_rms_norm=float(initial[j].norm(dim=-1).square().mean().sqrt()),
            final_rms_norm=float(books[j].norm(dim=-1).square().mean().sqrt()),
            lower_drift_rms=float((books[j,:,:64]-initial[j]).norm(dim=-1).square().mean().sqrt()),
            added_coordinates_rms_norm=float(books[j,:,64:].norm(dim=-1).square().mean().sqrt()))
    rng=np.random.default_rng(20261106);delta=[]
    diff=users['codebook_init','test']-users['control','test']
    for _ in range(50):delta.extend(diff[rng.integers(0,len(diff),size=(100,len(diff)))].mean(1).tolist())
    first_changes=[x['changed_items_previous']for x in indexrows['live_codebook'][1:]]
    last=stats(failed);assert last['max_collision_group']>256
    result=dict(quality_complete=False,complete_arms=['codebook_init'],failed_arms=['live_codebook'],failure_stage='start of epoch30 requantization after29completedtrainingepochs',
        failure=last,attempted_epoch30_without_added_coordinates=stats(lower),
        assignments_changed_by_added_coordinates=int(np.any(failed!=lower,axis=1).sum()),codebook_geometry=drift,
        verified_prediction_files=len(checks),curves=rows,index_curves=indexrows,
        live_churn_fraction_median=float(np.median(first_changes)/2874),
        test_control=float(users['control','test'].mean()),test_codebook_init=float(users['codebook_init','test'].mean()),
        test_relative_change=float(diff.mean()/users['control','test'].mean()),test_delta_conditional_user_ci95=np.quantile(delta,[.025,.975]).tolist(),
        bootstrap_scope='5000paireduserdraws,fixedmodels,onegeneratorseed;nottrainingseedinference',
        control_best_validation=load(parent/'trials/sum_seed42/training-selection.json')['validation_ndcg10'],
        codebook_init_best_validation=suite['results']['codebook_init']['selection']['validation_ndcg10'],
        codebook_init_training_seconds=suite['results']['codebook_init']['selection']['optimizer_training_seconds'],
        source_checkpoint_sha256=sha(checkpoint),no_new_training_or_generator_inference=True)
    for name,obj in [('summary',result),('verification',checks)]: (out/(name+'.json')).write_text(json.dumps(obj,indent=2)+'\n')
    plot(result,out);return result


def plot(s,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(2,2,figsize=(12,8))
    for a,color in [('control','#666666'),('codebook_init','#1b9e77'),('live_codebook','#d95f02')]:
        r=[x for x in s['curves']if x['arm']==a];ax[0,0].plot([x['epoch']for x in r],[x['ndcg10']for x in r],'-o',label=a,color=color)
    r=s['index_curves']['live_codebook'];e=[x['epoch']for x in r]
    for j in range(3):ax[0,1].plot(e,[x['used_codes'][j]for x in r],label=f'level{j+1}')
    ax[0,1].set(ylabel='Live index: occupied codes',ylim=(0,260))
    ax[1,0].plot(e,[x['changed_items_previous']/2874 for x in r],color='#d95f02');ax[1,0].set(ylabel='Fraction of full SIDs changed vs previous epoch',ylim=(0,1.05))
    ax[1,1].plot(e,[x['max_collision_group']for x in r],label='largest collision group')
    ax[1,1].scatter([30],[s['failure']['max_collision_group']],marker='x',s=70,label='failed epoch30 assignment')
    ax[1,1].axhline(256,color='red',ls='--',label='suffix capacity256');ax[1,1].set(ylabel='Items sharing all3semantic codes')
    ax[0,0].set(ylabel='VAL NDCG@10');
    for a in ax.flat:a.set(xlabel='Epoch');a.grid(alpha=.2)
    for a in [ax[0,0],ax[0,1],ax[1,1]]:a.legend(fontsize=8)
    fig.suptitle('Shared codebooks: static initialization completes; live reassignment collapses')
    fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(out/('shared-codebooks.'+ext),dpi=180)
    plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ['root','data','parent','checkpoint','output']:p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();s=replay(a.root,a.data,a.parent,a.checkpoint,a.output)
    print(json.dumps({k:v for k,v in s.items()if k not in ['curves','index_curves']},indent=2))
