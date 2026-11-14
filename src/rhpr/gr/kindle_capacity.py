"""Two wider generator fits; reuse the exact completed seed42 Kindle index."""
from __future__ import annotations
import copy, json, shutil, subprocess, sys
from pathlib import Path
from rhpr.gr.experiment import _write
from rhpr.gr import kindle_windows

TRIALS=[dict(id=f'recent{w}_wide_42',seed=42,history_items=w)for w in (20,32)]
WIDE=dict(d_model=128,d_ff=1024,d_kv=64)
NARROW=dict(d_model=64,d_ff=512,d_kv=32)


def validate(c):
    if (c['kind']!='gr_kindle_capacity' or c['trials']!=TRIALS or c['seeds']!=[42]
        or c['windows']!=[20,32] or any(c['generator'][k]!=v for k,v in WIDE.items())
        or c['parent']['kernel']!='carasik/rhpr-kindle-long-windows'
        or c['parent']['git_sha']!='55350fd1a3013a2673b1f9795b13c6e9d1efa0c7'
        or c['kaggle']['kernel_sources']!=[c['parent']['kernel']]):
        raise ValueError('capacity pilot differs from the two declared wide fits')
    normalized=copy.deepcopy(c)
    normalized.update(kind='gr_kindle_windows',trials=kindle_windows.TRIALS,seeds=kindle_windows.SEEDS,windows=kindle_windows.WINDOWS)
    normalized['generator'].update(NARROW)
    kindle_windows.validate(normalized)


def check_parent(c,previous):
    """Reject accidental data/optimization changes in this capacity comparison."""
    for key in ('data','movie_input','training','evaluation','quantizer'):
        if c[key]!=previous[key]:raise ValueError('capacity comparison changes '+key)
    expected={**previous['generator'],**WIDE}
    if c['generator']!=expected:raise ValueError('unexpected generator architecture change')


def run(c,output):
    import torch
    from rhpr.gr.ae_suite import assignments,launch_workers
    from rhpr.gr.data import file_sha256
    from rhpr.gr.grid import prepare_grid_source
    from rhpr.gr.pair_test import locate_parent
    from rhpr.gr.resources import unpack_grid,verify_resources
    from rhpr.offline_runtime import resolve_dataset_directory
    validate(c);output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True);(output/'workers').mkdir(exist_ok=True)
    if torch.cuda.device_count()!=2 or any(torch.cuda.get_device_capability(i)!=(7,5)for i in (0,1)):
        raise RuntimeError('capacity pilot requires two T4s; no fallback')
    torch.set_num_threads(1);parent=locate_parent(c['parent']);previous=json.loads((parent/'run-manifest.json').read_text())['config'];check_parent(c,previous)
    data=resolve_dataset_directory(c['movie_input']['path'])
    for name,digest in c['movie_input']['sha256'].items():
        if file_sha256(data/name)!=digest:raise ValueError('input digest mismatch: '+name)
    indexes=output/'frozen-indexes';indexes.mkdir();shutil.copy2(parent/'frozen-indexes/items-42.npy',indexes/'items-42.npy')
    baselines=output/'baselines';baselines.mkdir();rows={}
    for w in (20,32):
        name=f'recent{w}_42';destination=baselines/name;destination.mkdir()
        for file in c['parent']['files']:
            if file.startswith(f'trials/{name}/'):shutil.copy2(parent/file,destination/Path(file).name)
        rows[name]=json.loads((destination/'trial-result.json').read_text())
    _write(output/'baseline-results.json',rows)
    resources=resolve_dataset_directory(c['resources']['path']);verify_resources(resources,c['resources']['manifest_sha256'])
    modern=resolve_dataset_directory(c['modern_runtime']['path']);verify_resources(modern,c['modern_runtime']['manifest_sha256'])
    scratch=Path(c['resources']['scratch_path']);scratch.mkdir(parents=True,exist_ok=True)
    source=prepare_grid_source(unpack_grid(resources/'grid-source.bin',scratch/'original'),scratch/'grid')
    overlay=scratch/'modern-site';overlay.mkdir(parents=True,exist_ok=True)
    command=[sys.executable,'-m','pip','install','--no-index','--no-deps','--no-cache-dir','--no-compile','--require-hashes','--only-binary=:all:','--target',str(overlay),'--find-links',str(modern),'-r',str(modern/'requirements-lock.txt')]
    with (output/'modern-install.log').open('w')as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)
    plan=dict(config=c,output=str(output),source=str(source),data=str(data),indexes=str(indexes),assignments=assignments(c,2))
    _write(output/'plan.json',plan);sys.path.insert(0,str(overlay))
    # Reuse the already exercised worker/trial implementation and event contract.
    summary=launch_workers(output/'plan.json',worker_module='rhpr.gr.kindle_windows',event_prefix='kindle_windows')
    new=list(summary['results'].values())
    for key in ('initial_state_sha256','sid_file_sha256','raw_case_sha256','case_targets_sha256'):
        if len({json.dumps(r[key],sort_keys=True)for r in new})!=1:raise RuntimeError('unmatched wide '+key)
    for row in new:
        old=rows[f"recent{row['history_items']}_42"]
        for key in ('sid_file_sha256','raw_case_sha256','case_targets_sha256'):
            if row[key]!=old[key]:raise RuntimeError('wide vs baseline differs: '+key)
    _write(output/'study-summary.json',dict(results=summary['results'],baselines=rows,parent=c['parent'],test_evaluated=False,tokenizer_refitted=False))
    return dict(experiment=c['name'],models=2,reused_baselines=2,seed=42,test_evaluated=False)
