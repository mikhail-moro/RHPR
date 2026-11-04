"""Fixed-budget three-modality MTL screening, one pipeline seed."""
def validate(c):
    t=c['training'];q=c['quantizer']
    if (c['replicate_seeds']!=[42] or c['seed']!=42 or c['data']['history_items']!=20
        or t['max_epochs']!=30 or t['min_epochs']!=30 or t['validation_interval']!=5
        or t['patience_epochs']!=30 or 'scheduler' in t or t['validate_first_epoch']
        or t['precision']!='fp16' or t['evaluation_precision']!='fp32'
        or q['max_steps']!=3000 or q['ae_warmup']!={'max_steps':3500,'probe_interval':500,'stopping':{'kind':'fixed_steps'}}
        or q['codebook_size']!=256 or q['num_hierarchies']!=3
        or len(c['trials'])!=4 or {x['mtl_mode'] for x in c['trials']}!={'sum','uw','gradnorm','pcgrad'}
        or any(x['seed']!=42 or x['stage']!=3 or x['modalities']!=['text','poster','plot'] for x in c['trials'])):
        raise ValueError('MTL screen differs from declared protocol')
