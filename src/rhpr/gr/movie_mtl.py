"""Fixed-budget three-modality MTL screening, one pipeline seed."""
def validate(c):
    t=c['training'];q=c['quantizer']
    singleton=c['data']['split']=='latest_two_singleton_timestamp_groups'
    if c['data']['split'] not in {'latest_two_singleton_timestamp_groups','timestamp_group_leave_two'}:
        raise ValueError('unknown MTL temporal split')
    seeds=[42,101,202] if singleton else [42]
    if singleton and any(c['data'][k]!=v for k,v in dict(users=903,items=1273,train_events=82234,train_examples=80268).items()):
        raise ValueError('singleton screen needs pinned prepared counts')
    if set(c['expected_initial_state_sha256_by_seed'])!={str(s)for s in seeds} or set(q['expected_initial_state_sha256_by_seed'])!={str(s)for s in seeds}:
        raise ValueError('MTL needs per-seed generator and quantizer initialization pins')
    if (c['replicate_seeds']!=seeds or c['seed']!=42 or c['data']['history_items']!=20
        or t['max_epochs']!=30 or t['min_epochs']!=30 or t['validation_interval']!=5
        or t['patience_epochs']!=30 or 'scheduler' in t or t['validate_first_epoch']
        or t['precision']!='fp16' or t['evaluation_precision']!='fp32'
        or q['max_steps']!=3000 or q['ae_warmup']!={'max_steps':3500,'probe_interval':500,'stopping':{'kind':'fixed_steps'}}
        or q['codebook_size']!=256 or q['num_hierarchies']!=3
        or len(c['trials'])!=4*len(seeds) or {(x['seed'],x['mtl_mode']) for x in c['trials']}!={(s,m) for s in seeds for m in ['sum','uw','gradnorm','pcgrad']}
        or any(x['seed'] not in seeds or x['stage']!=3 or x['modalities']!=['text','poster','plot'] for x in c['trials'])):
        raise ValueError('MTL screen differs from declared protocol')
