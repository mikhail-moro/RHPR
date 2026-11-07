"""Two-seed, fixed30epoch screening of actor-only DeepWalk and loss weights."""
import copy,json
from pathlib import Path
import numpy as np
from rhpr.gr.data import file_sha256

CANDIDATES={'none':('none',1.),'shared_cast':('shared_graph',1.),
            'actor_w1':('actor_graph',1.),'actor_w03':('actor_graph',.3),'actor_w01':('actor_graph',.1)}


def validate(config):
    from rhpr.gr.movie_people_roles import validate as parent_validate
    t=config['training'];trials=config['trials']
    expected={(seed,name) for seed in [42,101] for name in CANDIDATES}
    if (config['replicate_seeds']!=[42,101] or len(trials)!=10
            or {(r['seed'],r['candidate']) for r in trials}!=expected
            or len({r['id'] for r in trials})!=10
            or t['max_epochs']!=30 or t['min_epochs']!=30 or t['patience_epochs']!=30
            or t['validation_interval']!=5 or 'scheduler' in t
            or t['validate_first_epoch'] or t['precision']!='fp16' or t['evaluation_precision']!='fp32'):
        raise ValueError('actor screen must use two seeds, five candidates and fixed30epochs/val5')
    for r in trials:
        source,weight=CANDIDATES[r['candidate']]
        active=['text','poster','plot']+([] if source=='none' else ['people'])
        if (r['people_source']!=source or r['people_role']!=('none' if source=='none' else 'cast')
                or r['modalities']!=active or r['stage']!=len(active)
                or r['reconstruction_weights']!=dict(text=1.,poster=1.,plot=1.,people=weight)):
            raise ValueError('actor screen source/role/weights differ from candidate matrix')
    for pins in [config['expected_initial_state_sha256_by_seed'],config['quantizer']['expected_initial_state_sha256_by_seed']]:
        if set(pins)!={'42','101'}:raise ValueError('actor screen requires two initialization pins')
    spec=config['actor_graph_input']
    if any(not isinstance(spec[k],str) or len(spec[k])!=64 or any(c not in '0123456789abcdef' for c in spec[k])
           for k in ['people_sha256','source_ids_sha256','provenance_sha256']):
        raise ValueError('actor graph input needs frozen hashes')
    parent=copy.deepcopy(config);parent['replicate_seeds']=[42,101,202]
    for pins in [parent['expected_initial_state_sha256_by_seed'],parent['quantizer']['expected_initial_state_sha256_by_seed']]:pins['202']=pins['42']
    parent['training'].update(max_epochs=100,min_epochs=30,patience_epochs=20,validation_interval=1,
        scheduler=dict(kind='reduce_on_plateau',factor=.5,patience_epochs=5,min_lr=1e-5,threshold=1e-4,start_epoch=21))
    parent['trials']=[dict(id=f'{role}_seed{seed}',seed=seed,people_role=role,stage=3 if role=='none' else 4,
        modalities=['text','poster','plot']+([] if role=='none' else ['people']),
        reconstruction_weights=dict(text=1.,poster=1.,plot=1.,people=1.))
        for seed in [42,101,202] for role in ['none','cast','directors']]
    parent_validate(parent)


def actor_features(spec,ids):
    from rhpr.offline_runtime import resolve_dataset_directory
    p=resolve_dataset_directory(spec['path'])
    for name,key in [('people.npy','people_sha256'),('source-ids.json','source_ids_sha256'),('people-provenance.json','provenance_sha256')]:
        if file_sha256(p/name)!=spec[key]:raise ValueError('actor-only graph input hash changed')
    if json.loads((p/'source-ids.json').read_text())!=ids:raise ValueError('actor graph catalogue order changed')
    provenance=json.loads((p/'people-provenance.json').read_text())
    if provenance['graph_role']!='cast' or not provenance['no_interactions_used'] or provenance['seed']!=42:
        raise ValueError('expected frozen actor-only metadata graph')
    x=np.load(p/'people.npy',allow_pickle=False)
    if x.shape!=(len(ids),128) or not np.isfinite(x).all() or np.any(x[:,64:]) or not np.any(x[:,:64]):
        raise ValueError('actor graph must contain cast64D and zero director padding')
    return x,provenance


def train_generator(model,users,index,config,output,device):
    from rhpr.gr.experiment import _train_generator
    from rhpr.gr.movie_data import temporal_examples
    result=_train_generator(model,users,index,config,output,device,example_builder=temporal_examples)
    if result['training_epochs']!=config['training']['max_epochs'] or result['validation_checks']!=config['training']['max_epochs']//config['training']['validation_interval']:
        raise RuntimeError('actor screening did not complete its fixed epoch/check schedule')
    return result
