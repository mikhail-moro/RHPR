"""One-seed clean-raw-window ML1M MTL comparison with native T5 SDPA."""
from __future__ import annotations
import argparse,copy,functools,gc,json,os,random,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
from rhpr.gr.experiment import _write,_event
from rhpr.gr.data import file_sha256
from rhpr.gr.movie_data import case_digest


def validate(c):
    t=c['training'];q=c['quantizer'];d=c['data']
    if (c['kind']!='gr_ml1m_clean_mtl' or c['seed']!=42 or c['replicate_seeds']!=[42]
            or d['split']!='raw_contiguous_20_val_test' or d['users']!=2786 or d['items']!=2874
            or d['train_examples']!=151166 or d['history_items']!=20
            or t['batch_size']!=512 or t['physical_batch_size']!=512
            or t['max_epochs']!=30 or t['min_epochs']!=30 or t['validation_interval']!=5
            or t['patience_epochs']!=30 or t['validate_first_epoch'] or 'scheduler'in t
            or t['precision']!='fp16' or t['evaluation_precision']!='fp32' or t['learning_rate']!=.001
            or c['generator']['attn_implementation']!='sdpa'
            or any(c['generator'][k]!=v for k,v in dict(d_model=128,d_ff=1024,d_kv=64,num_heads=6,encoder_layers=4,decoder_layers=4,dropout=.1,num_user_bins=2000,beam_width=20).items())
            or q['head_dim']!=128 or q['head_hidden']!=256
            or q['max_steps']!=3000 or q['init_buffer_size']!=2874
            or q['ae_warmup']!={'max_steps':3500,'probe_interval':500,'stopping':{'kind':'fixed_steps'}}
            or q['zero_inactive_slots']!=['people'] or q['num_hierarchies']!=3 or q['codebook_size']!=256
            or len(c['trials'])!=4 or {r['mtl_mode']for r in c['trials']}!={'sum','uw','gradnorm','pcgrad'}
            or any(r['seed']!=42 or r['modalities']!=['text','poster','plot']for r in c['trials'])):
        raise ValueError('ML1M clean MTL differs from the matched authorized protocol')
    for pins in [c['expected_initial_state_sha256_by_seed'],q['expected_initial_state_sha256_by_seed']]:
        if set(pins)!={'42'}or len(pins['42'])!=64:raise ValueError('single-seed initialization pins required')


@functools.lru_cache(maxsize=3)
def _cases(root,split):
    with np.load(Path(root)/(split+'-cases.npz'),allow_pickle=False)as z:
        result=tuple(z[k]for k in ['contexts','user_ids','targets'])
    for a in result:a.flags.writeable=False
    return result


def case_examples(root,history_items,split):
    if history_items!=20 or split not in {'train','validation','test'}:
        raise ValueError('clean cases require the declared20item window and split')
    return _cases(str(root),split)


def prepare_features(c,data,assets,resources,output,device):
    import torch
    from sentence_transformers import SentenceTransformer
    from rhpr.gr.movie_modal_features import prepare_shared_modalities
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    before=time.monotonic();texts=json.loads((data/'metadata-texts.json').read_text());ids=json.loads((data/'source-ids.json').read_text())
    model=SentenceTransformer(str(resources/'sentence-t5-base'),device=device,local_files_only=True)
    model.max_seq_length=2048;model.eval()
    lengths=[len(model.tokenizer.encode(s,add_special_tokens=True,truncation=False))for s in texts]
    if max(lengths)>2048:raise ValueError('metadata text exceeds pinned no-truncation policy')
    probe=model.tokenize([texts[int(np.argmax(lengths))]])['attention_mask']
    if int(probe.sum())!=max(lengths):raise ValueError('metadata tokenizer silently truncated input')
    text=model.encode(texts,batch_size=c['representation']['batch_size'],convert_to_numpy=True,show_progress_bar=False)
    if text.shape!=(2874,768)or not np.isfinite(text).all():raise ValueError('invalid metadata embeddings')
    del model;gc.collect();torch.cuda.empty_cache()if device.startswith('cuda')else None
    base=output/'base';base.mkdir(exist_ok=True)
    image=np.load(data/'poster.npy',allow_pickle=False)
    np.savez_compressed(base/'base-features.npz',text=text.astype(np.float32),poster=image,people=np.zeros((len(ids),128),dtype=np.float32),
        text_observed=np.ones(len(ids),dtype=bool),poster_observed=np.ones(len(ids),dtype=bool),people_observed=np.zeros(len(ids),dtype=bool))
    for name in ['source-ids.json','plot-texts.json']:(base/name).write_bytes((data/name).read_bytes())
    provenance=dict(items=len(ids),metadata_fields=c['fields'],metadata_encoder=c['representation'],metadata_tokens_max=max(lengths),metadata_truncated_items=0,
        poster='Zenodo15403972 json/vit_cls.json keyed by original ML1M ID',plot='Zenodo description ONLY; no genres/tags/reviews in plot channel',
        people='inactive zero compatibility slot; never enters encoder or reconstruction/MTL loss',no_user_interactions_used_for_features=True,input_manifest_sha256=c['movie_input']['manifest_sha256'])
    _write(base/'provenance.json',provenance)
    settings=copy.deepcopy(c);settings['plot_encoder']['path']=str(assets/'plot-encoder')
    root=prepare_shared_modalities(settings,base,resources,output/'shared-modalities',device)
    with np.load(root/'features.npz',allow_pickle=False)as z:
        if not z['plot_observed'].all()or z['plot'].shape!=(2874,768):raise ValueError('missing ML1M plot embedding')
    _event('ml1m_features_complete',seconds=time.monotonic()-before,features_sha256=file_sha256(root/'features.npz'))
    return root


def run_trial(trial,c,plan,device):
    import torch
    from rhpr.gr.multimodal_rqvae import fit_multimodal_rqvae
    from rhpr.gr.grid import build_grid_generator
    from rhpr.gr.sid import SidIndex
    from rhpr.gr.sid_study import state_digest
    from rhpr.gr.experiment import _train_generator,_evaluate
    root=Path(plan['output'])/'trials'/trial['id'];root.mkdir(parents=True,exist_ok=True)
    seed=42;random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    with np.load(Path(plan['modality_root'])/'features.npz',allow_pickle=False)as z:
        arrays={k:z[k]for k in ['text','poster','plot','people']};masks={k:z[k+'_observed']for k in arrays}
    if file_sha256(Path(plan['modality_root'])/'features.npz')!=plan['features_sha256']:
        raise ValueError('shared frozen embeddings changed')
    q=copy.deepcopy(c['quantizer']);q['mtl_mode']=trial['mtl_mode']
    _write(root/'representation.json',dict(active_modalities=trial['modalities'],frozen_sources=True,source_features_sha256=plan['features_sha256']))
    codes,report=fit_multimodal_rqvae(arrays,masks,q,seed,device,plan['source'],Path(plan['original'])/'configs/experiment/rqvae_train_flat.yaml',root,trial['modalities'])
    del arrays,masks;gc.collect();torch.cuda.empty_cache()if device.startswith('cuda')else None
    index=SidIndex(codes,width=c['quantizer']['codebook_size']);np.save(root/'semantic-ids.npy',index.sids,allow_pickle=False)
    _write(root/'sid-diagnostics.json',index.diagnostics());_write(root/'sid-catalogue.json',dict(original_item_ids=plan['source_ids']))
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    model=build_grid_generator(index.sids,c['generator'],seed,device,plan['source'])
    initial=state_digest(dict(model.named_parameters()))
    if initial!=c['expected_initial_state_sha256_by_seed']['42']:raise ValueError('fresh SDPA generator initialization changed')
    _write(root/'generator.json',dict(config=c['generator'],parameters=sum(p.numel()for p in model.parameters()),initial_state_sha256=initial,attention_backend='sdpa'))
    selection=_train_generator(model,plan['data_root'],index,c,root,device,example_builder=case_examples)
    if selection['training_epochs']!=30 or selection['validation_checks']!=6:raise RuntimeError('incomplete30epoch/val5fit')
    examples=case_examples(plan['data_root'],20,'test')
    metrics,pred,probs=_evaluate(model,examples,index,c['training']['evaluation_batch_size'],device,precision='fp32')
    np.savez_compressed(root/'test-predictions.npz',semantic_ids=pred,scores=probs,user_ids=examples[1],target_item_ids=examples[2])
    result=dict(trial=trial['id'],seed=seed,stage=3,mtl_mode=trial['mtl_mode'],active_modalities=trial['modalities'],initial_state_sha256=initial,
        quantizer_steps=report['steps'],selection=selection,test=metrics,features_sha256=plan['features_sha256'],case_sha256=c['data']['case_sha256'])
    _write(root/'trial-result.json',result);return result


def run_worker(plan_path,slot):
    import torch,transformers
    if transformers.__version__!='5.22.1':raise RuntimeError('ML1M workers require native SDPA runtime5.22.1')
    plan=json.loads(Path(plan_path).read_text());c=plan['config'];torch.set_num_threads(1)
    if torch.cuda.device_count()!=1 or torch.cuda.get_device_capability()!=(7,5):raise RuntimeError('each worker requires its assigned T4')
    status=dict(results={},failures=[]);job=plan['assignments'][slot]
    for trial in job['trials']:
        _event('ml1m_mtl_trial_start',trial=trial['id'],worker=slot)
        try:
            status['results'][trial['id']]=run_trial(trial,c,plan,'cuda:0')
            _event('ml1m_mtl_trial_complete',trial=trial['id'],result=status['results'][trial['id']])
        except Exception as exc:
            root=Path(plan['output'])/'trials'/trial['id'];root.mkdir(parents=True,exist_ok=True)
            (root/'failure.log').write_text(traceback.format_exc());failure=dict(trial=trial['id'],exception_type=type(exc).__name__,message=str(exc))
            status['failures'].append(failure);_event('ml1m_mtl_trial_failed',**failure)
        _write(Path(plan['output'])/'workers'/f'worker-{slot}.json',status);gc.collect();torch.cuda.empty_cache()
    _write(Path(plan['output'])/'workers'/f'worker-{slot}.json',status)


def run_clean_mtl(c,output):
    import torch
    from rhpr.offline_runtime import resolve_dataset_directory
    from rhpr.gr.resources import verify_resources,unpack_grid
    from rhpr.gr.grid import prepare_grid_source
    from rhpr.gr.ae_suite import assignments,launch_workers
    validate(c);output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True);(output/'workers').mkdir(exist_ok=True)
    if torch.cuda.device_count()<2 or any(torch.cuda.get_device_capability(i)!=(7,5)for i in [0,1]):raise RuntimeError('clean MTL requires2T4; no fallback')
    torch.set_num_threads(1)
    resources=resolve_dataset_directory(c['resources']['path']);verify_resources(resources,c['resources']['manifest_sha256'])
    data=resolve_dataset_directory(c['movie_input']['path']);verify_resources(data,c['movie_input']['manifest_sha256'])
    assets=resolve_dataset_directory(c['plot_assets']['path']);verify_resources(assets,c['plot_assets']['manifest_sha256'])
    modern=resolve_dataset_directory(c['modern_runtime']['path']);verify_resources(modern,c['modern_runtime']['manifest_sha256'])
    report=json.loads((data/'split-report.json').read_text())
    for k in ['items','users','train_examples','case_sha256']:
        if report[k]!=c['data'][k]:raise ValueError('prepared clean ML1M counts/hashes changed')
    for split in ['train','validation','test']:
        values=case_examples(data,20,split)
        if case_digest(values)!=c['data']['case_sha256'][split]:raise ValueError('clean ML1M case digest changed')
    scratch=Path(c['resources']['scratch_path']);original=unpack_grid(resources/'grid-source.bin',scratch/'original')
    source=prepare_grid_source(original,scratch/'grid')
    modal=prepare_features(c,data,assets,resources,output/'features','cuda:0')
    overlay=scratch/'modern-site';overlay.mkdir(parents=True,exist_ok=True)
    command=[sys.executable,'-m','pip','install','--no-index','--no-deps','--no-cache-dir','--no-compile','--require-hashes','--only-binary=:all:','--target',str(overlay),'--find-links',str(modern),'-r',str(modern/'requirements-lock.txt')]
    with(output/'modern-install.log').open('w')as f:subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=180)
    plan=dict(config=c,output=str(output),source=str(source),original=str(original),data_root=str(data),modality_root=str(modal),source_ids=json.loads((data/'source-ids.json').read_text()),features_sha256=file_sha256(modal/'features.npz'),assignments=assignments(c,torch.cuda.device_count()))
    _write(output/'plan.json',plan)
    # Parent retains its loaded legacy encoder runtime. Only fresh worker
    # interpreters see modern Transformers first in their inherited PYTHONPATH.
    sys.path.insert(0,str(overlay))
    summary=launch_workers(output/'plan.json',worker_module='rhpr.gr.ml1m_clean_mtl',event_prefix='ml1m_mtl')
    result=dict(experiment=c['name'],complete=len(summary['results']),users=2786,train_examples=151166)
    for trial,row in summary['results'].items():
        result[trial+'_validation_ndcg10']=row['selection']['validation_ndcg10'];result[trial+'_test_ndcg10']=row['test']['ndcg@10']
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--slot',type=int,required=True);a=p.parse_args();run_worker(a.plan,a.slot)
