"""Frozen TRAIN-only ALS channel and matched shuffled-vector control."""
import copy,json
from pathlib import Path
import numpy as np
from rhpr.gr.data import file_sha256
from rhpr.gr.resources import verify_resources


def validate_als(c):
    from rhpr.gr.ml1m_clean_mtl import validate
    if c['seed']!=42 or c['replicate_seeds']!=[42] or len(c['trials'])!=2:
        raise ValueError('ALS screen requires two seed42 sum trials')
    if c.get('cf_screen')=='objectives':
        if {(t.get('cf_variant'),t['mtl_mode'],t.get('cf_objective'))for t in c['trials']}!={('als','gradnorm','gradnorm'),('als','sum','no_cf_reconstruction')}:
            raise ValueError('objective screen requires ALS GradNorm and zero CF reconstruction')
    elif c.get('cf_screen','channel')=='channel':
        if {(t.get('cf_variant'),t['mtl_mode'])for t in c['trials']}!={('als','sum'),('shuffled','sum')}or any(t.get('cf_objective')for t in c['trials']):
            raise ValueError('both ALS and shuffled sum controls required')
    else:raise ValueError('unknown ALS screen')
    if any(t['seed']!=42 or t['modalities']!=['text','poster','plot','people']for t in c['trials']):
        raise ValueError('ALS screen requires four active input slots')
    if len({t['id']for t in c['trials']})!=2:raise ValueError('unique trials required')
    if len(c['collaborative_input']['manifest_sha256'])!=64:raise ValueError('ALS input must be pinned')
    base=copy.deepcopy(c);base.pop('collaborative_input')
    base['trials']=[dict(id=m+'_seed42',seed=42,mtl_mode=m,modalities=['text','poster','plot'])for m in ['sum','uw','gradnorm','pcgrad']]
    validate(base)


def configure_objective(spec,trial):
    """Keep all input heads active; zero only the fixed CF reconstruction weight."""
    if trial.get('cf_objective')=='no_cf_reconstruction':
        # The MTL adapter overrides fixed reconstruction weights, so the fixed
        # weighted sum uses the existing native path without that adapter.
        spec.pop('mtl_mode',None)
        spec['reconstruction_weights']={'people':0.}
    return spec


def attach_channel(arrays,masks,root,variant,manifest_sha256):
    root=Path(root);verify_resources(root,manifest_sha256)
    if variant not in {'als','shuffled'}:raise ValueError('unknown collaborative variant')
    values=np.load(root/(variant+'.npy'),allow_pickle=False);observed=np.load(root/'observed.npy',allow_pickle=False)
    if values.shape!=arrays['people'].shape or observed.shape!=(len(values),) or observed.dtype!=bool:
        raise ValueError('ALS slot shape/mask mismatch')
    if not np.isfinite(values).all()or not observed.any()or values[~observed].any():raise ValueError('invalid ALS factors')
    arrays['people']=values;masks['people']=observed
    return dict(kind='TRAIN-only ALS',variant=variant,internal_slot='people',contains_people_data=False,
                factors_sha256=file_sha256(root/(variant+'.npy')),manifest_sha256=manifest_sha256,
                observed_items=int(observed.sum()),diagnostics=json.loads((root/'als-report.json').read_text()))
