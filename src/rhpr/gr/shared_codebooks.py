"""TIGER codebook initialization versus live epoch-wise RQ reassignment."""
from __future__ import annotations
import argparse,copy,json,random,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
from rhpr.gr.experiment import _write,_event
from rhpr.gr.data import file_sha256


def assign(latents,books):
    import torch
    with torch.no_grad():
        residual=latents.float().clone();codes=[]
        for b in books:
            # Native squared Euclidean residual quantization, no normalization.
            j=torch.cdist(residual,b.float(),p=2).square().argmin(-1)
            codes.append(j);residual=residual-b[j].float()
        return torch.stack(codes,1).cpu().numpy()


def initialize(model,books):
    import torch
    table=model.item_sid_embedding_table_encoder.weight
    books=torch.as_tensor(books,device=table.device,dtype=table.dtype)
    if books.shape!=(3,256,64) or table.shape!=(1024,128):raise ValueError('unexpected codebook/table shapes')
    with torch.no_grad():table[:768].copy_(torch.nn.functional.pad(books,(0,64)).reshape(768,128))
    return table


def install(model,index,latents,dynamic,output):
    import torch
    from rhpr.gr.sid import SidIndex
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    latents=torch.nn.functional.pad(torch.as_tensor(latents,device=next(model.parameters()).device),(0,64))
    model.register_buffer('rhpr_active_sids',torch.as_tensor(index.sids.copy(),device=latents.device))
    initial=index.sids.copy();rows=[]
    def restore(idx):
        restored=SidIndex.from_full_sids(model.rhpr_active_sids.detach().cpu().numpy(),width=256)
        idx.__dict__.update(restored.__dict__)
        model.codebooks=model.rhpr_active_sids
    def start(epoch,idx,root):
        before=time.monotonic();previous=idx.sids.copy()
        books=model.item_sid_embedding_table_encoder.weight[:768].reshape(3,256,128)
        if dynamic:
            codes=assign(latents,books)
            try:
                new=SidIndex(codes,width=256)
            except ValueError as error:
                # Preserve the failed candidate before suffix construction aborts.
                np.save(output/f'failed-semantic-codes-epoch-{epoch:03d}.npy',codes,allow_pickle=False)
                np.save(output/f'failed-codebooks-epoch-{epoch:03d}.npy',books.detach().float().cpu().numpy(),allow_pickle=False)
                _,counts=np.unique(codes,axis=0,return_counts=True)
                _write(output/'index-failure.json',dict(epoch=epoch,stage='epoch_start_reassignment',
                    message=str(error),max_collision_group=int(counts.max()),
                    used_codes=[int(len(np.unique(codes[:,k])))for k in range(3)],
                    overflow_groups=int((counts>256).sum()),items_in_overflow_groups=int(counts[counts>256].sum())))
                raise
            model.rhpr_active_sids.copy_(torch.as_tensor(new.sids.copy(),device=latents.device))
            restore(idx)
        if epoch==1 and not np.array_equal(idx.sids,initial):raise ValueError('initial requantization changes original SIDs')
        np.save(output/f'semantic-ids-epoch-{epoch:03d}.npy',idx.sids,allow_pickle=False)
        row=dict(epoch=epoch,changed_items_previous=int(np.any(idx.sids!=previous,axis=1).sum()),
            changed_items_initial=int(np.any(idx.sids!=initial,axis=1).sum()),
            used_codes=[int(len(np.unique(idx.sids[:,k])))for k in range(3)],
            max_collision_group=idx.diagnostics()['max_collision_group'],seconds=time.monotonic()-before)
        rows.append(row);_write(output/'index-learning-curves.json',rows);_event('shared_codebook_index',**row)
    model.rhpr_epoch_start=start;model.rhpr_after_checkpoint_load=restore
    restore(index)


def validate(c):
    if c.get('kind')=='gr_codebook_adaptation':
        from rhpr.gr.codebook_adaptation import validate as adaptation_validate
        return adaptation_validate(c)
    t=c['training'];g=c['generator']
    if (c['kind']!='gr_shared_codebooks' or c['seed']!=42 or len(c['trials'])!=2
        or [(r['id'],r['dynamic'])for r in c['trials']]!=[('codebook_init',False),('live_codebook',True)]
        or any(r['seed']!=42 for r in c['trials']) or t['max_epochs']!=30 or t['min_epochs']!=30
        or t['validation_interval']!=5 or t['patience_epochs']!=30 or t['validate_first_epoch']
        or t['batch_size']!=512 or t['physical_batch_size']!=512 or t['precision']!='fp16'
        or t['evaluation_precision']!='fp32' or t['learning_rate']!=.001 or 'scheduler'in t
        or c['data']['train_examples']!=151166 or c['data']['history_items']!=20
        or c['evaluation']['test_after_selection'] is not True
        or any(g[k]!=v for k,v in dict(d_model=128,d_ff=1024,d_kv=64,num_heads=6,encoder_layers=4,decoder_layers=4,dropout=.1,attn_implementation='sdpa').items())):
        raise ValueError('shared-codebook experiment differs from authorized two-fit protocol')


def frozen_latents(c,parent,source,output):
    """Load fixed RQ checkpoint once; no fitting or holdout reads."""
    import torch
    from rhpr.gr.rqvae import build_rqvae
    from rhpr.gr.multimodal_rqvae import normalize_modalities
    from rhpr.gr.modality_mtl import ModalityMTL
    cp=parent/'trials/sum_seed42/quantizer.pt'
    if file_sha256(cp)!=c['parent_files']['quantizer.pt']:raise ValueError('quantizer checkpoint changed')
    ck=torch.load(cp,map_location='cpu',weights_only=False)
    from rhpr.gr.grid import load_grid
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    load_grid(source)
    q=instantiate(OmegaConf.create(ck['resolved_model']))
    q.mtl=ModalityMTL('sum',ck['active_modalities'],42);q.load_state_dict(ck['model'],strict=True);q.eval()
    for p in q.parameters():p.requires_grad_(False)
    fp=parent/'features/shared-modalities/features.npz'
    if file_sha256(fp)!=c['parent_files']['features.npz']:raise ValueError('frozen content features changed')
    with np.load(fp,allow_pickle=False)as z:
        v={k:z[k]for k in ck['schema']};m={k:z[k+'_observed']for k in ck['schema']}
    a,_=normalize_modalities({k:v[k]for k in ck['active_modalities']},{k:m[k]for k in ck['active_modalities']})
    a['people']=np.zeros_like(v['people']);f={k:torch.tensor(x)for k,x in a.items()}
    f.update({k+'_observed':torch.tensor(x)for k,x in m.items()})
    # Match native export batch shape and FP32 eval exactly.
    with torch.no_grad():lat=torch.cat([q.encode_modalities({k:v[j:j+2048]for k,v in f.items()})for j in range(0,2874,2048)]).numpy()
    np.save(output/'frozen-latents.npy',lat,allow_pickle=False)
    books=np.load(parent/'trials/sum_seed42/codebooks-final.npz',allow_pickle=False)['codebooks']
    np.testing.assert_array_equal(assign(torch.from_numpy(lat),torch.from_numpy(books)),np.load(parent/'trials/sum_seed42/semantic-codes.npy'))
    np.save(output/'initial-books.npy',books,allow_pickle=False)
    return lat


def run_trial(trial,c,plan,device):
    import torch
    from rhpr.gr.grid import build_grid_generator
    from rhpr.gr.sid import SidIndex
    from rhpr.gr.sid_study import state_digest
    from rhpr.gr.ml1m_clean_mtl import case_examples
    from rhpr.gr.experiment import _train_generator,_evaluate
    seed=42;random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    root=Path(plan['output'])/'trials'/trial['id'];root.mkdir(parents=True,exist_ok=True)
    index=SidIndex.from_full_sids(np.load(Path(plan['assets'])/'initial-sids.npy'),width=256)
    model=build_grid_generator(index.sids,c['generator'],seed,device,plan['source'])
    initialize(model,np.load(Path(plan['assets'])/'initial-books.npy'))
    install(model,index,np.load(Path(plan['assets'])/'frozen-latents.npy'),trial['dynamic'],root)
    pin=state_digest(dict(model.named_parameters()))
    if pin!=c['initialization_pin']:raise ValueError('initial model changed')
    _write(root/'generator.json',dict(initial_state_sha256=pin,parameters=sum(p.numel()for p in model.parameters()),
        config=c['generator'],dynamic_sid=trial['dynamic'],shared_rows=768,zero_lift='64 ->128',rq_loss=False))
    selection=_train_generator(model,plan['data_root'],index,c,root,device,example_builder=case_examples)
    if selection['training_epochs']!=30 or selection['validation_checks']!=6:raise RuntimeError('incomplete training')
    # The best checkpoint stores its matching active SID snapshot.
    np.save(root/'semantic-ids.npy',index.sids,allow_pickle=False)
    metrics,pred,scores=_evaluate(model,case_examples(plan['data_root'],20,'test'),index,512,device,'fp32')
    cases=case_examples(plan['data_root'],20,'test')
    np.savez_compressed(root/'test-predictions.npz',semantic_ids=pred,scores=scores,user_ids=cases[1],target_item_ids=cases[2])
    row=dict(trial=trial['id'],seed=seed,initial_state_sha256=pin,selection=selection,test=metrics)
    _write(root/'trial-result.json',row);return row


def worker(path,slot):
    import torch,transformers
    p=json.loads(Path(path).read_text());c=p['config'];torch.set_num_threads(1)
    if torch.cuda.device_count()!=1 or torch.cuda.get_device_capability()!=(7,5)or transformers.__version__!='5.22.1':raise RuntimeError('requires native5.22.1 and oneT4 per worker')
    trial=p['assignments'][slot]['trials'][0];status=dict(results={},failures=[])
    _event('shared_codebook_trial_start',trial=trial['id'],worker=slot,dynamic_sid=trial['dynamic'])
    try:
        trial_runner=run_trial
        if c.get('kind')=='gr_codebook_adaptation':
            from rhpr.gr.codebook_adaptation import run_trial as trial_runner
        status['results'][trial['id']]=trial_runner(trial,c,p,'cuda:0')
        _event('shared_codebook_trial_complete',trial=trial['id'],worker=slot)
    except Exception as e:
        _event('shared_codebook_trial_failed',trial=trial['id'],worker=slot,exception_type=type(e).__name__,message=str(e))
        status['failures'].append(dict(trial=trial['id'],exception_type=type(e).__name__,message=str(e)));(Path(p['output'])/f'worker-{slot}-failure.log').write_text(traceback.format_exc())
    _write(Path(p['output'])/'workers'/f'worker-{slot}.json',status)


def run(c,output):
    import torch
    from rhpr.gr.resources import verify_resources,verify_parent_artifacts,unpack_grid
    from rhpr.gr.grid import prepare_grid_source
    from rhpr.gr.ae_suite import assignments,launch_workers
    from rhpr.gr.ml1m_clean_mtl import case_examples
    from rhpr.gr.movie_data import case_digest
    from rhpr.offline_runtime import resolve_dataset_directory
    validate(c);output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True);(output/'workers').mkdir(exist_ok=True)
    if torch.cuda.device_count()!=2 or any(torch.cuda.get_device_capability(i)!=(7,5)for i in [0,1]):raise RuntimeError('requires2T4')
    roots={k:resolve_dataset_directory(c[k]['path'])for k in ['resources','movie_input','modern_runtime']}
    for k,v in roots.items():verify_resources(v,c[k]['manifest_sha256'])
    for split in ['train','validation','test']:
        if case_digest(case_examples(roots['movie_input'],20,split))!=c['data']['case_sha256'][split]:raise ValueError('case arrays changed')
    parent=verify_parent_artifacts(c['parent'])
    for f in ['codebooks-final.npz','semantic-codes.npy','semantic-ids.npy']:
        if file_sha256(parent/'trials/sum_seed42'/f)!=c['parent_files'][f]:raise ValueError('parent codebook/SID changed')
    scratch=Path(c['resources']['scratch_path'])
    original=unpack_grid(roots['resources']/'grid-source.bin',scratch/'original');source=prepare_grid_source(original,scratch/'grid')
    assets=output/'frozen-quantizer';assets.mkdir();frozen_latents(c,parent,source,assets)
    np.save(assets/'initial-sids.npy',np.load(parent/'trials/sum_seed42/semantic-ids.npy'))
    if c.get('kind')=='gr_codebook_adaptation':
        from rhpr.gr.codebook_adaptation import stage
        stage(c,parent,assets)
    overlay=scratch/'modern-site';overlay.mkdir(parents=True,exist_ok=True);modern=roots['modern_runtime']
    with (output/'modern-install.log').open('w')as log:
        subprocess.run([sys.executable,'-m','pip','install','--no-index','--no-deps','--no-cache-dir','--no-compile','--require-hashes','--only-binary=:all:','--target',str(overlay),'--find-links',str(modern),'-r',str(modern/'requirements-lock.txt')],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)
    plan=dict(config=c,output=str(output),assets=str(assets),source=str(source),data_root=str(roots['movie_input']),assignments=assignments(c,2))
    _write(output/'plan.json',plan);sys.path.insert(0,str(overlay))
    result=launch_workers(output/'plan.json',worker_module='rhpr.gr.shared_codebooks',event_prefix='shared_codebook')
    return {**{f"{k}_test_ndcg10":r['test']['ndcg@10']for k,r in result['results'].items()},'completed':2}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--slot',required=True,type=int);a=p.parse_args();worker(a.plan,a.slot)
