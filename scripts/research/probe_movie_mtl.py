import json,sys,torch,numpy as np
from pathlib import Path
from rhpr.gr.grid import prepare_grid_source,build_grid_generator
from rhpr.gr.movie_multimodal import run_multimodal_trial
from rhpr.gr.movie_data import TemporalUser
from rhpr.gr.sid_study import state_digest
root=Path(sys.argv[1]);root.mkdir(exist_ok=True);original=Path(sys.argv[2]);source=prepare_grid_source(original,root/'grid')
torch.set_num_threads(1)
c=json.loads(Path('experiments/movielens100k_mtl_2t4.json').read_text())
# Full-sized original initializer before adding task-balancing parameters.
from rhpr.gr.rqvae import build_rqvae
torch.manual_seed(42)
full,_=build_rqvae(source,original/'configs/experiment/rqvae_train_flat.yaml',512,c['quantizer'],modality_schema=dict(text=768,poster=768,plot=768,people=128),active_modalities=['text','poster','plot'])
assert state_digest(dict(full.named_parameters()))==c['quantizer']['expected_initial_state_sha256_by_seed']['42'];del full
c['quantizer']=dict(num_hierarchies=3,codebook_size=16,batch_size=16,max_steps=6,init_buffer_size=16,initialization_steps=1,distance_backend='torch_cdist',latent_dim=4,head_dim=4,head_hidden=8,diagnostics_interval_steps=3,ae_warmup=dict(max_steps=3,probe_interval=3,stopping=dict(kind='fixed_steps')))
c['generator'].update(d_model=8,d_ff=16,d_kv=4,num_heads=2,encoder_layers=1,decoder_layers=1,dropout=0.,num_user_bins=0,codebook_size=16,beam_width=2)
c['training'].update(device='cpu',precision='fp32',evaluation_precision='fp32',amp=False,batch_size=8,physical_batch_size=8,evaluation_batch_size=2,max_seconds=120)
data=root/'features';data.mkdir(exist_ok=True);dims=dict(text=8,poster=6,plot=5,people=4)
a={k:np.random.default_rng(d).normal(size=(16,d)).astype('float32') for k,d in dims.items()}
np.savez_compressed(data/'features.npz',**a,**{k+'_observed':np.ones(16,dtype=bool) for k in dims})
(data/'source-ids.json').write_text(json.dumps(list(range(16))));(data/'provenance.json').write_text('{}')
users=[TemporalUser(1,(0,1,2,3),(1,1,2,2),4,3,5,4),TemporalUser(2,(6,7,8,9),(1,1,2,2),10,3,11,4)]
dummy=np.zeros((16,4),dtype=np.int64);dummy[:,0]=np.arange(16)//16;dummy[:,1]=np.arange(16)%16
m=build_grid_generator(dummy,c['generator'],42,'cpu',source);initial=state_digest(dict(m.named_parameters()));del m
plan=dict(output=str(root/'suite'),modality_root=str(data),source_ids=list(range(16)),original=str(original),initial_state_sha256_by_seed={'42':initial})
for t in c['trials']:
 r=run_multimodal_trial(t,c,plan,users,source,'cpu')
 assert r['selection']['training_epochs']==30 and r['selection']['validation_checks']==6
 print('E2E_OK',t['mtl_mode'],flush=True)
