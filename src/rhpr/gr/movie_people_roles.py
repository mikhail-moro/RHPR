"""Matched actor/director input and reconstruction ablation, frozen DeepWalk."""
import copy


def role_columns(role):
    if role == 'none':
        return None
    if role not in {'cast', 'directors'}:
        raise ValueError('unknown people role')
    return list(range(0, 64) if role == 'cast' else range(64, 128))


def validate(config):
    from rhpr.gr.movie_graph_loss import validate as validate_parent
    trials = config['trials']
    expected = {(s, r) for s in [42, 101, 202] for r in ['none', 'cast', 'directors']}
    pairs = {(t['seed'], t['people_role']) for t in trials}
    if (config['replicate_seeds'] != [42, 101, 202] or len(trials) != 9 or pairs != expected
            or len({t['id'] for t in trials}) != 9
            or 'reconstruction_columns' in config['quantizer']):
        raise ValueError('people role ablation requires three matched roles and three seeds')
    for t in trials:
        active = ['text', 'poster', 'plot'] + ([] if t['people_role'] == 'none' else ['people'])
        if (t['modalities'] != active or t['stage'] != len(active)
                or t['reconstruction_weights'] != dict(text=1., poster=1., plot=1., people=1.)):
            raise ValueError('role inputs/losses differ from the declared ablation')
    for pins in [config['expected_initial_state_sha256_by_seed'], config['quantizer']['expected_initial_state_sha256_by_seed']]:
        if set(pins) != {'42', '101', '202'} or any(len(v) != 64 or any(c not in '0123456789abcdef' for c in v) for v in pins.values()):
            raise ValueError('all three initializations must be pinned')
    # Reuse the strict fixed-warmup/full-training protocol contract, replacing
    # only the old trial matrix and its single-seed pin declarations.
    parent = copy.deepcopy(config)
    parent['replicate_seeds'] = [42]
    for pins in [parent['expected_initial_state_sha256_by_seed'], parent['quantizer']['expected_initial_state_sha256_by_seed']]:
        pins.pop('101'); pins.pop('202')
    parent['trials'] = [dict(id=name, seed=42, stage=4,
        modalities=['text','poster','plot','people'], reconstruction_weights=dict(text=1.,poster=1.,plot=1.,people=w))
        for name,w in [('graph_weight_1_seed42',1.),('graph_weight_01_seed42',.1)]]
    validate_parent(parent)
