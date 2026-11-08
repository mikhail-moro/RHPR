"""Paired graph reconstruction weighting with fixed AE and full TIGER training."""
import copy
import hashlib
import shutil
from pathlib import Path


def validate(config):
    from rhpr.gr.movie_multimodal import MODALITIES
    q=config['quantizer'];t=config['training']
    weights={r['id']:r.get('reconstruction_weights') for r in config['trials']}
    expected={'graph_weight_1_seed42':dict(text=1.,poster=1.,plot=1.,people=1.),
              'graph_weight_01_seed42':dict(text=1.,poster=1.,plot=1.,people=.1)}
    if (config['seed']!=42 or config['replicate_seeds']!=[42] or len(config['trials'])!=2 or weights!=expected
            or any(r['seed']!=42 or r['stage']!=4 or r['modalities']!=list(MODALITIES) for r in config['trials'])
            or config['compare_initialization_by_seed'] is not True
            or config['data']['items']!=1273 or config['data']['users']!=943 or config['data']['history_items']!=20
            or config['data']['split']!='timestamp_group_leave_two'
            or q['kind']!='grid_multimodal_rqvae' or q['num_hierarchies']!=3 or q['codebook_size']!=256
            or q['max_steps']!=3000 or q['head_dim']!=128 or q['head_hidden']!=256
            or q['ae_warmup']!={'max_steps':3500,'probe_interval':500,'stopping':{'kind':'fixed_steps'}}
            or 'reconstruction_weights' in q or q['init_buffer_size']!=1273
            or t['min_epochs']!=30 or t['max_epochs']!=100 or t['patience_epochs']!=20
            or t['validation_interval']!=1 or t['validate_first_epoch']
            or t['precision']!='fp16' or t['evaluation_precision']!='fp32'
            or t['scheduler']!={'kind':'reduce_on_plateau','factor':.5,'patience_epochs':5,'min_lr':1e-5,'threshold':1e-4,'start_epoch':21}
            or config['modality_input']['features_sha256']!='eeccc0bf7e11f0e2d97acce14ce4973b96d0b4816c6e35ffef679681802dabda'
            or config['fields']!=['title_with_year','genres','explicit_year','runtime','tmdb_keywords']
            or not config['evaluation']['save_validation_losses']):
        raise ValueError('graph loss pilot differs from paired weights1/.1 fixed-warmup seed42 protocol')
    for pins in [q['expected_initial_state_sha256_by_seed'],config['expected_initial_state_sha256_by_seed']]:
        if set(pins)!={'42'} or any(not isinstance(h,str) or len(h)!=64 or any(c not in '0123456789abcdef' for c in h) for h in pins.values()):
            raise ValueError('graph loss pilot needs pinned seed42 initialization')


def train_generator(model,users,index,config,output,device):
    from rhpr.gr.experiment import _train_generator,_write
    from rhpr.gr.movie_data import temporal_examples
    output=Path(output)
    prefix=copy.deepcopy(config);prefix['training'].update(max_epochs=20,min_epochs=20,validation_interval=5)
    prefix['training'].pop('scheduler',None)
    first=_train_generator(model,users,index,prefix,output,device,example_builder=temporal_examples)
    if first['training_epochs']!=20 or first['validation_checks']!=4:
        raise RuntimeError('graph loss prefix did not complete20epochs and4validation checks')
    for name in ['training-curves.json','training-selection.json']:
        shutil.copy2(output/name,output/('prefix-'+name))
    _write(output/'prefix-checkpoint-provenance.json',dict(epoch=20,
        checkpoint_sha256=hashlib.sha256((output/'generator-last.pt').read_bytes()).hexdigest(),
        validation_epochs=[5,10,15,20],test_evaluated=False))
    selected=_train_generator(model,users,index,config,output,device,example_builder=temporal_examples,resume_from=output)
    if selected['stopping_reason']=='time_budget':
        raise RuntimeError('graph loss full training exhausted time guard; no automatic extension')
    return selected
