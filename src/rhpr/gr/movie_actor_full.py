"""Frozen validation-selected actor graph weight .1, full paired training."""
import copy


def validate(config):
    from rhpr.gr.movie_people_roles import validate as parent_validate
    trials=config['trials']
    expected={(seed,name) for seed in [42,101,202] for name in ['none','actor_w01']}
    if (config['replicate_seeds']!=[42,101,202] or len(trials)!=6
            or {(t['seed'],t['candidate']) for t in trials}!=expected
            or len({t['id'] for t in trials})!=6):
        raise ValueError('full actor comparison requires two candidates on three seeds')
    for t in trials:
        none=t['candidate']=='none';active=['text','poster','plot']+([] if none else ['people'])
        if (t['people_source']!=('none' if none else 'actor_graph')
                or t['people_role']!=('none' if none else 'cast')
                or t['stage']!=len(active) or t['modalities']!=active
                or t['reconstruction_weights']!=dict(text=1.,poster=1.,plot=1.,people=1. if none else .1)):
            raise ValueError('full comparison must freeze actor-only graph and selected weight .1')
    expected_hashes=dict(people_sha256='93fadcb9eda245b4c34562d237555b0851037b46d84f793ea05d4a1965a8758f',
        source_ids_sha256='9c4a806b3ed1aa818e9fcdbfd92fd20d6adbbe05c522c4086a8d463f17e0f136',
        provenance_sha256='65b13bdfe86a7ea8f15020c79187eb868efd17e22f96de920e13e1334a32fbee')
    if any(config['actor_graph_input'].get(k)!=v for k,v in expected_hashes.items()):
        raise ValueError('actor graph inputs differ from the selected screen')
    parent=copy.deepcopy(config)
    parent['trials']=[dict(id=f'{role}_seed{seed}',seed=seed,people_role=role,
        stage=3 if role=='none' else 4,modalities=['text','poster','plot']+([] if role=='none' else ['people']),
        reconstruction_weights=dict(text=1.,poster=1.,plot=1.,people=1.))
        for seed in [42,101,202] for role in ['none','cast','directors']]
    parent_validate(parent)
