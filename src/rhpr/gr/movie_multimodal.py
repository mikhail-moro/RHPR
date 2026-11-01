"""Movie metadata/poster/plot/people ladder with a shared residual bottleneck."""
import gc,json,random,time
from pathlib import Path
import numpy as np
from rhpr.gr.experiment import _write
from rhpr.gr.movie_data import temporal_examples
from rhpr.gr.data import file_sha256

MODALITIES=('text','poster','plot','people')


def validate_modalities(config):
    q=config['quantizer']
    if (config['fields']!=['title_with_year','genres','explicit_year','runtime','tmdb_keywords']
            or q['kind']!='grid_multimodal_rqvae' or q['num_hierarchies']!=3
            or q['codebook_size']!=256 or q['max_steps']!=3000
            or q['head_dim']!=128 or q['head_hidden']!=256
            or config['replicate_seeds']!=[42,101,202]
            or config.get('plot_encoder',{}).get('model')!='sentence-transformers/all-mpnet-base-v2'
            or config['plot_encoder']['max_length']!=384
            or any(t['modalities']!=list(MODALITIES[:t['stage']]) for t in config['trials'])):
        raise ValueError('multimodal ladder differs from declared matched protocol')


def run_multimodal_trial(trial, config, plan, users, source, device):
    import torch
    from rhpr.gr.multimodal_rqvae import fit_multimodal_rqvae
    from rhpr.gr.grid import build_grid_generator
    from rhpr.gr.sid import SidIndex
    from rhpr.gr.sid_study import state_digest
    from rhpr.gr.experiment import _train_generator,_evaluate
    seed=trial['seed'];config={**config,'seed':seed};root=Path(plan['output'])/'trials'/trial['id'];root.mkdir(parents=True,exist_ok=True)
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    data=Path(plan['modality_root'])
    ids=json.loads((data/'source-ids.json').read_text())
    if ids!=plan['source_ids']:raise ValueError('modality catalogue order differs from temporal cases')
    with np.load(data/'features.npz',allow_pickle=False) as z:
        arrays={k:z[k] for k in MODALITIES};masks={k:z[k+'_observed'] for k in MODALITIES}
    if any(len(v)!=len(ids) for v in arrays.values()):raise ValueError('wrong modality catalogue size')
    _write(root/'representation.json',{'active_modalities':trial['modalities'],'frozen_sources':True,
        'source_features_sha256':file_sha256(data/'features.npz'),'provenance':json.loads((data/'provenance.json').read_text())})
    codes,report=fit_multimodal_rqvae(arrays,masks,config['quantizer'],seed,device,source,
        Path(plan['original'])/'configs/experiment/rqvae_train_flat.yaml',root,trial['modalities'])
    del arrays;gc.collect()
    if device.startswith('cuda'):torch.cuda.empty_cache()
    index=SidIndex(codes,width=config['quantizer']['codebook_size'])
    np.save(root/'semantic-ids.npy',index.sids,allow_pickle=False);_write(root/'sid-diagnostics.json',index.diagnostics())
    _write(root/'sid-catalogue.json',{'original_item_ids':ids})
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    model=build_grid_generator(index.sids,config['generator'],seed,device,source)
    initial=state_digest(dict(model.named_parameters()))
    if initial!=plan['initial_state_sha256_by_seed'][str(seed)]:raise ValueError('generator initial weights differ within seed')
    _write(root/'generator.json',{'parameters':sum(p.numel() for p in model.parameters()),'initial_state_sha256':initial,'config':config['generator']})
    selection=_train_generator(model,users,index,config,root,device,example_builder=temporal_examples)
    if selection['training_epochs']!=20 or selection['validation_checks']!=4:raise RuntimeError('incomplete20epoch multimodal fit')
    examples=temporal_examples(users,config['data']['history_items'],'test')
    metrics,predictions,scores=_evaluate(model,examples,index,config['training']['evaluation_batch_size'],device,precision='fp32')
    np.savez_compressed(root/'test-predictions.npz',semantic_ids=predictions,scores=scores,user_ids=examples[1],target_item_ids=examples[2])
    result={'trial':trial['id'],'stage':trial['stage'],'seed':seed,'initial_state_sha256':initial,
        'active_modalities':trial['modalities'],'quantizer_steps':report['steps'],'selection':selection,'test':metrics}
    _write(root/'trial-result.json',result);del model;gc.collect()
    return result
