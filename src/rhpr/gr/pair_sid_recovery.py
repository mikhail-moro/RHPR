"""Complete seed-42 pair SID fits after the original parent occupied a GPU."""
from __future__ import annotations
import copy,gc,json,subprocess,sys
from pathlib import Path
import numpy as np
from rhpr.gr.experiment import _event,_write
from rhpr.gr.movie_data import case_digest
from rhpr.gr.ml1m_clean_mtl import case_examples
from rhpr.gr.pair_sid import validate as validate_pilot


def validate(c):
    if c['kind']!='gr_ml1m_pair_sid_recovery':raise ValueError('wrong recovery kind')
    if c['trials'] != [dict(id='control_42',mode='control',seed=42),dict(id='pair_42',mode='pair',seed=42)]:raise ValueError('recovery must contain exactly the two missing fits')
    original=copy.deepcopy(c);original['kind']='gr_ml1m_pair_sid';validate_pilot(original)
    if c['recovery_input']['source_commit']!='22d06b2d528291de0cabcd7a9c2f6e8c8ff5356d':raise ValueError('wrong parent provenance')


def run(c,output):
    import torch
    from rhpr.gr.ae_suite import assignments,launch_workers
    from rhpr.gr.grid import prepare_grid_source
    from rhpr.gr.resources import unpack_grid,verify_resources
    from rhpr.gr.sid import SidIndex
    from rhpr.offline_runtime import resolve_dataset_directory
    validate(c);output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True);(output/'workers').mkdir(exist_ok=True)
    if torch.cuda.device_count()!=2 or any(torch.cuda.get_device_capability(i)!=(7,5) for i in (0,1)):raise RuntimeError('requires two T4 GPUs')
    torch.set_num_threads(1);roots={}
    for name in ('resources','movie_input','modern_runtime','recovery_input'):
        roots[name]=resolve_dataset_directory(c[name]['path']);verify_resources(roots[name],c[name]['manifest_sha256'])
    for split in ('train','validation'):
        if case_digest(case_examples(roots['movie_input'],20,split))!=c['data']['case_sha256'][split]:raise ValueError('cases differ from original run')
    root=roots['recovery_input'];items=np.load(root/'items-42.npy');keys=np.load(root/'pair-keys.npy');pairs=np.load(root/'pairs-42.npy')
    idx=SidIndex.from_full_sids(items,width=256)
    if items.shape!=(2874,4) or keys.shape!=(124812,) or pairs.shape!=(124812,3) or np.any(keys[1:]<=keys[:-1]) or np.any(pairs<0) or np.any(pairs>=256):raise ValueError('invalid retained indexes')
    source_record=json.loads((root/'source.json').read_text())
    if source_record['source_commit']!=c['recovery_input']['source_commit']:raise ValueError('index source mismatch')
    scratch=Path(c['resources']['scratch_path']);scratch.mkdir(parents=True,exist_ok=True)
    source=prepare_grid_source(unpack_grid(roots['resources']/'grid-source.bin',scratch/'original'),scratch/'grid')
    overlay=scratch/'modern-site';overlay.mkdir(parents=True,exist_ok=True)
    command=[sys.executable,'-m','pip','install','--no-index','--no-deps','--no-cache-dir','--no-compile','--require-hashes','--only-binary=:all:',
        '--target',str(overlay),'--find-links',str(roots['modern_runtime']),'-r',str(roots['modern_runtime']/'requirements-lock.txt')]
    with (output/'modern-install.log').open('w') as f:subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=180)
    gc.collect();torch.cuda.empty_cache()
    gpu_memory=[torch.cuda.mem_get_info(i)[0] for i in (0,1)]
    _event('pair_recovery_gpu_ready',free_bytes=gpu_memory)
    if min(gpu_memory)<8*1024**3:raise RuntimeError('insufficient free GPU memory before workers; refusing partial run')
    plan_config=copy.deepcopy(c);plan_config['kind']='gr_ml1m_pair_sid'
    plan=dict(config=plan_config,output=str(output),source=str(source),data_root=str(roots['movie_input']),index_root=str(root),assignments=assignments(plan_config,2))
    _write(output/'plan.json',plan);sys.path.insert(0,str(overlay));_event('pair_recovery_start',source=source_record,trials=2)
    summary=launch_workers(output/'plan.json',worker_module='rhpr.gr.pair_sid',event_prefix='pair')
    records=summary['results']
    if set(records)!={'control_42','pair_42'} or records['control_42']['initial_state_sha256']!=records['pair_42']['initial_state_sha256']:
        raise RuntimeError('recovery incomplete or generator initializations differ')
    _write(output/'recovery-summary.json',dict(results=records,source=source_record,test_evaluated=False))
    return dict(experiment=c['name'],models=len(records),seed=42,test_evaluated=False,reused_original_indexes=True)
