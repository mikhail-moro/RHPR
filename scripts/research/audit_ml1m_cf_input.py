import argparse,hashlib,json
from pathlib import Path
import numpy as np,torch
from rhpr.gr.rqvae import build_rqvae
from rhpr.gr.multimodal_rqvae import normalize_modalities
from rhpr.gr.sid_study import state_digest
parser=argparse.ArgumentParser()
for name in ['root','checkpoint','cf','grid_source','recipe','output']:parser.add_argument(name,type=Path)
a=parser.parse_args()
root=a.root;trial=root/'trials/als_no_cf_reconstruction_seed42';q= json.loads((trial/'quantizer.json').read_text());path=a.checkpoint
assert hashlib.sha256(path.read_bytes()).hexdigest()==q['checkpoint_sha256']
state=torch.load(path,map_location='cpu',weights_only=False);torch.set_num_threads(1);torch.manual_seed(42)
model,_=build_rqvae(a.grid_source,a.recipe,512,state['spec'],modality_schema=state['schema'],active_modalities=state['active_modalities'])
assert state_digest(dict(model.named_parameters()))==q['quantizer_initial_state_sha256']
initial={k:v.clone()for k,v in model.state_dict().items()if k.startswith(('encoder_heads.people.','decoder_heads.people.'))}
from rhpr.gr.multimodal_warmup import NEURAL_MODULES
for name in NEURAL_MODULES:getattr(model,name).load_state_dict({k[len(name)+1:]:v for k,v in state['model'].items()if k.startswith(name+'.')})
final=model.state_dict();encoder_delta=max(float((final[k]-v).abs().max())for k,v in initial.items()if k.startswith('encoder_heads'));decoder_delta=max(float((final[k]-v).abs().max())for k,v in initial.items()if k.startswith('decoder_heads'))
assert encoder_delta>0 and decoder_delta==0
with np.load(root/'features/shared-modalities/features.npz')as z:
 values={k:z[k]for k in ['text','poster','plot','people']};masks={k:z[k+'_observed']for k in values}
values['people']=np.load(a.cf/'als.npy');masks['people']=np.load(a.cf/'observed.npy');arrays,_=normalize_modalities(values,masks)
fields={k:torch.from_numpy(v)for k,v in arrays.items()};fields.update({k+'_observed':torch.from_numpy(v)for k,v in masks.items()});model.eval()
with torch.inference_mode():
 z=model.encode_modalities(fields);muted=dict(fields);muted['people_observed']=torch.zeros_like(fields['people_observed']);z0=model.encode_modalities(muted)
 observed=fields['people_observed'];relative=(z[observed]-z0[observed]).norm(dim=1)/z[observed].norm(dim=1).clamp_min(1e-12)
report=dict(checkpoint_sha256=q['checkpoint_sha256'],cf_encoder_max_parameter_change=encoder_delta,cf_decoder_max_parameter_change=decoder_delta,observed_items=int(observed.sum()),cf_block_masking_changes_latent_fraction=float(((z[observed]-z0[observed]).abs().amax(1)>1e-6).float().mean()),relative_latent_l2_change_median=float(relative.median()),relative_latent_l2_change_p95=float(torch.quantile(relative,.95)),scope='CPU tokenizer-only read-only probe, mask CF block after its MLP. Demonstrates surviving input sensitivity, not useful CF semantics, generator behavior or a ranking improvement. No generator inference/training.')
a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
