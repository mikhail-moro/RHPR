"""Frozen native codebooks with projection versus fully trainable shared RQ/TIGER."""
from __future__ import annotations
import copy,json,math,random,shutil,time
from pathlib import Path
import numpy as np
from rhpr.gr.experiment import _write,_event


def validate(c):
    from rhpr.gr.shared_codebooks import validate as old_validate
    old=copy.deepcopy(c);old['kind']='gr_shared_codebooks';old['trials']=[dict(id='codebook_init',seed=42,dynamic=False),dict(id='live_codebook',seed=42,dynamic=True)]
    old_validate(old)
    if ([(t['id'],t['dynamic'])for t in c['trials']]!=[('projected_frozen',False),('joint_rq',True)]
        or c['auxiliary']!=dict(batch_size=512,reconstruction_weight=1.,quantization_weight=1.,joint_latent_dim=128)
        or c['training']['max_seconds']!=2050 or set(c['initialization_pins'])!={'projected_frozen','joint_rq'}
        or any(len(v)!=64 or any(ch not in '0123456789abcdef'for ch in v)for v in c['initialization_pins'].values())):
        raise ValueError('adaptation differs from the authorized two-arm40minute screen')


def stage(c,parent,assets):
    import torch
    from rhpr.gr.multimodal_rqvae import normalize_modalities
    ck=torch.load(parent/'trials/sum_seed42/quantizer.pt',map_location='cpu',weights_only=False)
    with np.load(parent/'features/shared-modalities/features.npz',allow_pickle=False)as z:
        values={k:z[k]for k in ck['schema']};masks={k:z[k+'_observed']for k in ck['schema']}
    arrays,_=normalize_modalities({k:values[k]for k in ck['active_modalities']},{k:masks[k]for k in ck['active_modalities']})
    arrays['people']=np.zeros_like(values['people']);arrays.update({k+'_observed':v for k,v in masks.items()})
    np.savez_compressed(assets/'normalized-fields.npz',**arrays)
    shutil.copyfile(parent/'trials/sum_seed42/quantizer.pt',assets/'quantizer-checkpoint.pt')


def load_quantizer(assets,source):
    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from rhpr.gr.grid import load_grid
    from rhpr.gr.modality_mtl import ModalityMTL
    load_grid(source);ck=torch.load(Path(assets)/'quantizer-checkpoint.pt',map_location='cpu',weights_only=False)
    q=instantiate(OmegaConf.create(ck['resolved_model']));q.mtl=ModalityMTL('sum',ck['active_modalities'],42)
    q.load_state_dict(ck['model'],strict=True)
    if ck['steps']!=3000:raise ValueError('expected trained RQ checkpoint')
    for layer in q.quantization_layer_list:
        # Native initialization flags are Python attributes, absent from state_dict.
        layer.is_initialized=True;layer.is_initial_step=False
    return q


def widen_quantizer(q):
    """Function-preserving64->128latent expansion; no bridge/projection in this arm."""
    import torch
    from torch import nn
    last=max(i for i,m in enumerate(q.encoder.model)if isinstance(m,nn.Linear))
    enc=q.encoder.model[last];dec=q.decoder.model[0]
    if enc.out_features!=64 or dec.in_features!=64:raise ValueError('unexpected native MLP endpoints')
    with torch.random.fork_rng(devices=[]):
        newenc=nn.Linear(enc.in_features,128,bias=enc.bias is not None)
        newdec=nn.Linear(128,dec.out_features,bias=dec.bias is not None)
    with torch.no_grad():
        newenc.weight.zero_();newenc.weight[:64].copy_(enc.weight)
        if enc.bias is not None:newenc.bias.zero_();newenc.bias[:64].copy_(enc.bias)
        newdec.weight.zero_();newdec.weight[:,:64].copy_(dec.weight)
        if dec.bias is not None:newdec.bias.copy_(dec.bias)
    q.encoder.model[last]=newenc;q.decoder.model[0]=newdec
    q.encoder.output_dim=128;q.decoder.input_dim=128
    for layer in q.quantization_layer_list:
        layer.centroids=nn.Parameter(torch.nn.functional.pad(layer.centroids.detach(),(0,64)))
        layer.n_features=128
    return q


def direct_lookup(q,original_weight):
    import torch
    from torch import nn
    class DirectLookup(nn.Module):
        def __init__(self):
            super().__init__();self.quantizer=q
            self.suffix=nn.Parameter(original_weight[768:].detach().clone())
            self.num_embeddings=1024;self.embedding_dim=128
        @property
        def weight(self):return torch.cat([l.centroids for l in self.quantizer.quantization_layer_list]+[self.suffix],dim=0)
        def forward(self,indices):return torch.nn.functional.embedding(indices,self.weight)
    return DirectLookup()


def encode_catalogue(q,fields):
    import torch
    modes={m:m.training for m in q.modules()};q.eval()
    try:
        with torch.no_grad():return torch.cat([q.encode_modalities({k:v[j:j+2048]for k,v in fields.items()})for j in range(0,len(next(iter(fields.values()))),2048)])
    finally:
        for m,mode in modes.items():m.training=mode


def auxiliary_loss(q,fields):
    """Use GRID's actual STE strategy and beta loss, without Lightning initialization."""
    import torch
    encoded=q.encode_modalities(fields);residual=encoded;total=encoded.new_zeros(());quantized=torch.zeros_like(encoded)
    for layer in q.quantization_layer_list:
        # The native layer.model_step does this exact sequence for initialized books.
        _,vectors,ste=layer.quantization_strategy.quantize(codebook=layer.centroids,batch=residual)
        total=total+layer.loss_function(residual,vectors)
        quantized=quantized+ste;residual=residual-ste
    reconstruction,_=q.reconstruction(quantized,fields)
    return reconstruction,total


def install(model,index,fields,joint,output,steps_per_epoch):
    import torch
    from rhpr.gr.sid import SidIndex
    from rhpr.gr.shared_codebooks import assign
    output=Path(output);q=model.item_sid_embedding_table_encoder.quantizer
    model.register_buffer('rhpr_active_sids',torch.as_tensor(index.sids.copy(),device=next(model.parameters()).device))
    initial=index.sids.copy();trace=[];loss_rows=[];sums=np.zeros(3);updates=0;epoch_number=0
    rng=np.random.default_rng(424242)
    def restore(idx):
        idx.__dict__.update(SidIndex.from_full_sids(model.rhpr_active_sids.detach().cpu().numpy()).__dict__)
        model.codebooks=model.rhpr_active_sids
    def start(epoch,idx,root):
        nonlocal epoch_number
        epoch_number=epoch;previous=idx.sids.copy();before=time.monotonic()
        if joint:
            z=encode_catalogue(q,fields);books=torch.stack([l.centroids for l in q.quantization_layer_list])
            codes=assign(z,books)
            try:new=SidIndex(codes)
            except ValueError as e:
                np.save(output/f'failed-semantic-codes-epoch-{epoch:03d}.npy',codes)
                np.save(output/f'failed-codebooks-epoch-{epoch:03d}.npy',books.detach().cpu().numpy())
                np.save(output/f'failed-latents-epoch-{epoch:03d}.npy',z.detach().cpu().numpy())
                _,counts=np.unique(codes,axis=0,return_counts=True)
                _write(output/'index-failure.json',dict(epoch=epoch,message=str(e),max_collision_group=int(counts.max())))
                raise
            if epoch==1 and not np.array_equal(new.sids,initial):raise ValueError('initial expansion changed SIDs')
            model.rhpr_active_sids.copy_(torch.as_tensor(new.sids.copy(),device=z.device));restore(idx)
        np.save(output/f'semantic-ids-epoch-{epoch:03d}.npy',idx.sids)
        row=dict(epoch=epoch,changed_items_previous=int(np.any(idx.sids!=previous,axis=1).sum()),
            used_codes=[len(np.unique(idx.sids[:,j]))for j in range(3)],max_collision_group=idx.diagnostics()['max_collision_group'],seconds=time.monotonic()-before)
        trace.append(row);_write(output/'index-learning-curves.json',trace);_event('codebook_adaptation_index',**row)
    def loss(batch):
        nonlocal updates,sums
        hidden=model(**batch)[:,:-1]
        nip=sum(torch.nn.functional.cross_entropy(head(hidden[:,j]),batch['future_ids'][:,j])for j,head in enumerate(model.decoder.decoder_mlp))
        rec=nip.new_zeros((),dtype=torch.float32);vq=rec
        if joint and model.training and torch.is_grad_enabled():
            indices=torch.as_tensor(rng.integers(0,len(next(iter(fields.values()))),size=512),device=hidden.device)
            with torch.autocast(device_type=hidden.device.type,enabled=False):
                rec,vq=auxiliary_loss(q,{k:v[indices]for k,v in fields.items()})
        if model.training and torch.is_grad_enabled():
            sums+=np.array([nip.detach().float().item(),rec.detach().float().item(),vq.detach().float().item()]);updates+=1
            if updates%steps_per_epoch==0:
                loss_rows.append(dict(epoch=epoch_number,nip=sums[0]/steps_per_epoch,reconstruction=sums[1]/steps_per_epoch,quantization=sums[2]/steps_per_epoch))
                _write(output/'objective-learning-curves.json',loss_rows);sums[:]=0
        return nip+rec+vq
    model.rhpr_epoch_start=start;model.rhpr_after_checkpoint_load=restore;model.rhpr_loss=loss;restore(index)


def build(trial,c,plan,device):
    import torch
    from rhpr.gr.grid import build_grid_generator
    from rhpr.gr.projected_codebooks import projected_lookup
    from rhpr.gr.sid import SidIndex
    from rhpr.gr.sid_study import state_digest
    assets=Path(plan['assets']);q=load_quantizer(assets,plan['source'])
    joint=trial['dynamic']
    if joint:widen_quantizer(q)
    q.requires_grad_(joint);q.to(device)
    random.seed(42);np.random.seed(42);torch.manual_seed(42)
    index=SidIndex.from_full_sids(np.load(assets/'initial-sids.npy'))
    model=build_grid_generator(index.sids,c['generator'],42,device,plan['source'])
    base=state_digest(dict(model.named_parameters()))
    if base!='d57fc2e783c0a04bfc567c0082541f757e241d1c53d7e79b3e8f281bf2eb9337':raise ValueError('base TIGER initialization changed')
    original=model.item_sid_embedding_table_encoder.weight
    model.item_sid_embedding_table_encoder=(direct_lookup(q,original)if joint else projected_lookup(q,original))
    with np.load(assets/'normalized-fields.npz')as z:fields={k:torch.as_tensor(z[k],device=device)for k in z.files}
    return model,index,fields


def run_trial(trial,c,plan,device):
    import torch
    from rhpr.gr.experiment import _train_generator,_evaluate
    from rhpr.gr.ml1m_clean_mtl import case_examples
    from rhpr.gr.sid_study import state_digest
    root=Path(plan['output'])/'trials'/trial['id'];root.mkdir(parents=True,exist_ok=True)
    model,index,fields=build(trial,c,plan,device)
    pin=state_digest(dict(model.named_parameters()))
    if pin!=c['initialization_pins'][trial['id']]:raise ValueError('adaptation initial weights changed')
    install(model,index,fields,trial['dynamic'],root,math.ceil(len(case_examples(plan['data_root'],20,'train')[2])/c['training']['batch_size']))
    _write(root/'generator.json',dict(config=c['generator'],initial_state_sha256=pin,parameters=sum(p.numel()for p in model.parameters()),
        trainable_parameters=sum(p.numel()for p in model.parameters()if p.requires_grad),joint_rq=trial['dynamic'],rq_latent_dim=128 if trial['dynamic']else 64))
    selection=_train_generator(model,plan['data_root'],index,c,root,device,example_builder=case_examples)
    if selection['training_epochs']!=30 or selection['validation_checks']!=6:raise RuntimeError('incomplete30epochfit within budget')
    np.save(root/'semantic-ids.npy',index.sids)
    cases=case_examples(plan['data_root'],20,'test');metric,pred,scores=_evaluate(model,cases,index,512,device,'fp32')
    np.savez_compressed(root/'test-predictions.npz',semantic_ids=pred,scores=scores,user_ids=cases[1],target_item_ids=cases[2])
    row=dict(trial=trial['id'],seed=42,initial_state_sha256=pin,initialization_group=trial['id'],selection=selection,test=metric)
    _write(root/'trial-result.json',row);return row
