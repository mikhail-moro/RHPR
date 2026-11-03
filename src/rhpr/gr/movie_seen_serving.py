"""Frozen MovieLens predictors: eligibility filtering with common beam refill.

Raw GRID probabilities are unchanged; no training or checkpoint reselection.
"""
import argparse,gc,json,time
from pathlib import Path
import numpy as np
from rhpr.gr.experiment import _event,_write


def known_sets(users,case_users,split):
    if split not in {'validation','test'}:raise ValueError('heldout split required')
    lookup={u.user_id:u for u in users}
    return [set(lookup[int(u)].train)|({lookup[int(u)].validation}if split=='test'else set())for u in case_users]


def eligible_topk(decoded,scores,known,k=10):
    if decoded.shape!=scores.shape or len(decoded)!=len(known):raise ValueError('prediction/mask shape mismatch')
    items=np.full((len(decoded),k),-1,dtype=np.int64);values=np.full((len(decoded),k),-1.,dtype=np.float32)
    for j,row in enumerate(decoded):
        used=set();at=0
        for i,s in zip(row,scores[j]):
            i=int(i)
            if i<0 or i in known[j] or i in used:continue
            used.add(i);items[j,at]=i;values[j,at]=s;at+=1
            if at==k:break
    return items,values


def retrieval_metrics(items,targets):
    hit=items==targets[:,None];rank=np.where(hit.any(1),hit.argmax(1)+1,0)
    return {f'{name}@{k}':float(np.mean(np.where((rank>0)&(rank<=k),1/np.log2(np.maximum(rank,1)+1)if name=='ndcg'else 1.,0.)))for k in [5,10]for name in ['ndcg','recall']}


def decode_paths(paths,sids):
    lu={tuple(row):i for i,row in enumerate(sids)}
    return np.asarray([[lu.get(tuple(row),-1)for row in sample]for sample in paths],dtype=np.int64)


def generate(model,examples,index,device,batch_size,width):
    import torch
    from rhpr.gr.experiment import _index_model_batch
    model.top_k_for_generation=width;paths=[];scores=[];model.eval()
    with torch.inference_mode():
        for start in range(0,len(examples[2]),batch_size):
            b=_index_model_batch(*(x[start:start+batch_size]for x in examples),index,device)
            p,s=model.generate(input_ids=b['input_ids'],attention_mask=b['attention_mask_encoder'],user_id=b['user_id'])
            if not torch.isfinite(s).all():raise FloatingPointError('nonfinite serving beam scores')
            paths.append(p.cpu().numpy());scores.append(s.cpu().numpy())
    return np.concatenate(paths),np.concatenate(scores)


def exact_one(model,index,examples,j,device,batch_size=64):
    """Exhaust valid full paths only for users still short after maximum beam."""
    import torch
    from rhpr.gr.experiment import _index_model_batch
    batch=_index_model_batch(*(x[j:j+1]for x in examples),index,device)
    result=[]
    with torch.inference_mode():
        enc,mask=model.encoder_forward_pass(attention_mask=batch['attention_mask_encoder'],input_ids=batch['input_ids'],user_id=batch['user_id'])
        for start in range(0,len(index.sids),batch_size):
            sids=torch.as_tensor(index.sids[start:start+batch_size],device=device);n=len(sids)
            hidden=model.decoder_forward_pass(future_ids=sids[:,:-1],encoder_output=enc.repeat_interleave(n,0),attention_mask_for_encoder=mask.repeat_interleave(n,0),use_cache=False)
            if isinstance(hidden,tuple):hidden=hidden[0]
            logs=torch.zeros(n,device=device)
            for level in range(sids.shape[1]):
                logits=model.decoder.decoder_mlp[level](hidden[:,level,:]).float()
                logs+=torch.log_softmax(logits,-1).gather(1,sids[:,level:level+1]).squeeze(1)
            result.append(logs.cpu().numpy())
    return np.concatenate(result)


def run_trial(trial,config,plan,users,device):
    import torch
    from rhpr.gr.grid import build_grid_generator
    from rhpr.gr.sid import SidIndex
    from rhpr.gr.sid_study import state_digest
    from rhpr.gr.movie_data import temporal_examples,case_digest
    source=Path(plan['parent'])/'trials'/trial['id'];out=Path(plan['output'])/'trials'/trial['id'];out.mkdir(parents=True,exist_ok=True)
    checkpoint=torch.load(source/'generator-best.pt',map_location='cpu',weights_only=False)
    if checkpoint['epoch']!=trial['best_epoch']:raise ValueError('selected epoch changed')
    c=checkpoint['config'];index=SidIndex(np.load(source/'semantic-codes.npy'),width=c['generator']['codebook_size'])
    np.testing.assert_array_equal(index.sids,np.load(source/'semantic-ids.npy'))
    model=build_grid_generator(index.sids,c['generator'],trial['seed'],device,plan['source']);model.load_state_dict(checkpoint['model'],strict=True);del checkpoint;gc.collect()
    before=state_digest(dict(model.named_parameters()));result=dict(trial=trial['id'],seed=trial['seed'],candidate=trial['candidate'],checkpoint_epoch=trial['best_epoch'],splits={})
    for split in ['validation','test']:
        examples=temporal_examples(users,20,split)
        if case_digest(examples)!=plan['case_hashes'][split]:raise ValueError('serving cases changed')
        known=known_sets(users,examples[1],split)
        if any(int(t)in k for t,k in zip(examples[2],known)):raise ValueError('target was already known')
        cached=source/('test-predictions.npz'if split=='test'else f"validation-epoch-{trial['best_epoch']:03d}.npz")
        with np.load(cached)as z:
            np.testing.assert_array_equal(z['user_ids'],examples[1]);np.testing.assert_array_equal(z['target_item_ids'],examples[2]);old=z['semantic_ids'];old_scores=z['scores']
        # Same original32-user batch for a numerical replay anchor.
        anchor=tuple(x[:32]for x in examples);p,s=generate(model,anchor,index,device,32,20)
        if not np.array_equal(p,old[:32]):raise RuntimeError('parent beam20 anchor differs; do not interpret serving deltas')
        np.testing.assert_allclose(s,old_scores[:32],rtol=2e-4,atol=1e-8)
        started=time.monotonic();paths,scores=generate(model,examples,index,device,config['evaluation']['batch_size'],100);decoded=decode_paths(paths,index.sids)
        items,values=eligible_topk(decoded,scores,known);short=np.flatnonzero((items<0).any(1));refill_count=len(short)
        if len(short):
            subset=tuple(x[short]for x in examples);p2,s2=generate(model,subset,index,device,config['evaluation']['batch_size'],256)
            d2=decode_paths(p2,index.sids);v2,q2=eligible_topk(d2,s2,[known[j]for j in short]);items[short]=v2;values[short]=q2
            np.savez_compressed(out/f'{split}-refill256.npz',semantic_ids=p2,scores=s2,user_ids=examples[1][short],target_item_ids=examples[2][short])
        exact=np.flatnonzero((items<0).any(1))
        for j in exact:
            logs=exact_one(model,index,examples,int(j),device);order=np.argsort(-logs,kind='stable');allowed=[i for i in order if i not in known[j]][:10]
            items[j]=allowed;values[j]=np.exp(logs[allowed]);np.savez_compressed(out/f'{split}-exact-user-{int(examples[1][j])}.npz',log_probabilities=logs)
        if (items<0).any()or any(any(int(i)in k for i in row)for row,k in zip(items,known)):raise RuntimeError('incomplete or ineligible serving top10')
        np.savez_compressed(out/f'{split}-beam100.npz',semantic_ids=paths,scores=scores,user_ids=examples[1],target_item_ids=examples[2])
        np.savez_compressed(out/f'{split}-eligible-top10.npz',item_ids=items,scores=values,user_ids=examples[1],target_item_ids=examples[2])
        row=dict(original_beam20=retrieval_metrics(decode_paths(old,index.sids),examples[2]),unmasked_beam100=retrieval_metrics(decoded,examples[2]),eligible_refilled=retrieval_metrics(items,examples[2]),refill256_users=refill_count,exact_fallback_users=len(exact),known_outputs=0,users=len(items),seconds=time.monotonic()-started,anchor_beam20_exact=True)
        expected=json.loads((source/'trial-result.json').read_text());ref=expected['test']['ndcg@10']if split=='test'else expected['selection']['validation_ndcg10']
        if abs(row['original_beam20']['ndcg@10']-ref)>1e-12:raise ValueError('original metric reconstruction differs')
        result['splits'][split]=row;_write(out/'result.json',result);_event('movie_seen_split_complete',trial=trial['id'],split=split,metrics=row)
    if state_digest(dict(model.named_parameters()))!=before:raise RuntimeError('serving mutated learned weights')
    result.update(weights_unchanged=True,index_retrained=False,generator_retrained=False)
    _write(out/'result.json',result);return result


def run_worker(plan_path,slot):
    import torch
    from rhpr.gr.movie_data import load_temporal_users
    plan=json.loads(Path(plan_path).read_text());torch.set_num_threads(1)
    if torch.cuda.device_count()!=1:raise RuntimeError('worker requires its assigned one visible GPU')
    users=load_temporal_users(plan['sequences'],plan['source_ids']);status=dict(results={},failures=[])
    for trial in plan['assignments'][slot]['trials']:
        _event('movie_seen_trial_start',trial=trial['id'],worker=slot)
        status['results'][trial['id']]=run_trial(trial,plan['config'],plan,users,'cuda:0')
        _write(Path(plan['output'])/'workers'/f'worker-{slot}.json',status);gc.collect();torch.cuda.empty_cache()
        _event('movie_seen_trial_complete',trial=trial['id'])


def run_serving(config,output):
    import torch
    from rhpr.gr.resources import verify_parent_artifacts,unpack_grid
    from rhpr.gr.grid import prepare_grid_source
    from rhpr.gr.data import file_sha256
    from rhpr.gr.ae_suite import assignments,launch_workers
    from rhpr.offline_runtime import resolve_dataset_directory
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True);(output/'workers').mkdir(exist_ok=True)
    if config['evaluation']!={'batch_size':16,'precision':'fp32','beam_width':100,'refill_width':256,'cutoff':10}:raise ValueError('serving policy differs')
    if {(t['candidate'],t['seed'])for t in config['trials']}!={(c,s)for c in ['none','actor_w01']for s in [42,101,202]}or len(config['trials'])!=6:raise ValueError('must reuse six fixed pipelines')
    parent=verify_parent_artifacts(config['parent']);pc=json.loads((parent/'run-manifest.json').read_text())['config']
    for trial in config['trials']:
        source=parent/'trials'/trial['id']
        if trial['id']!=f"{trial['candidate']}_seed{trial['seed']}":raise ValueError('unsafe trial identity')
        for name,digest in trial['files'].items():
            if Path(name).name!=name or file_sha256(source/name)!=digest:raise ValueError('parent checkpoint or SID/prediction bytes changed')
    data=resolve_dataset_directory(config['movie_input']['path']);seq=data/'sequences.json'
    if file_sha256(seq)!=config['movie_input']['sequences_sha256']:raise ValueError('sequences changed')
    resources=resolve_dataset_directory(config['resources']['path']);archive=resources/'grid-source.bin'
    manifest=resources/'resources-manifest.json'
    if file_sha256(manifest)!=pc['resources']['manifest_sha256']:raise ValueError('resource manifest changed')
    entries=json.loads(manifest.read_text())['files'];entry=next(x for x in entries if x['path']=='grid-source.bin')
    if file_sha256(archive)!=entry['sha256']:raise ValueError('GRID archive changed')
    scratch=Path('/kaggle/temp/rhpr-movie-seen');original=unpack_grid(archive,scratch/'original');source=prepare_grid_source(original,scratch/'runtime')
    ids=json.loads((parent/'trials'/config['trials'][0]['id']/'sid-catalogue.json').read_text())['original_item_ids']
    if any(json.loads((parent/'trials'/t['id']/'sid-catalogue.json').read_text())['original_item_ids']!=ids for t in config['trials']):raise ValueError('catalogue order differs')
    plan=dict(config=config,parent=str(parent),source=str(source),sequences=str(seq),source_ids=ids,case_hashes=json.loads((parent/'split-report.json').read_text())['case_sha256'],output=str(output),assignments=assignments(config,torch.cuda.device_count()))
    _write(output/'worker-plan.json',plan);_event('movie_seen_ready',models=6,training=False,beam100=True,refill256=True)
    summary=launch_workers(output/'worker-plan.json',worker_module=__name__,event_prefix='movie_seen');metrics={}
    for split in ['validation','test']:
        for cand in ['none','actor_w01']:
            rows=[r['splits'][split]for r in summary['results'].values()if r['candidate']==cand]
            for policy in ['original_beam20','unmasked_beam100','eligible_refilled']:
                metrics[f'{split}_{cand}_{policy}_ndcg10']=float(np.mean([r[policy]['ndcg@10']for r in rows]))
    metrics.update(training_updates=0.,models_evaluated=6.);return metrics


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--slot',type=int,required=True);a=p.parse_args();run_worker(a.plan,a.slot)
