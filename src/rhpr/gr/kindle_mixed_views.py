"""One Full32/Pair32 batch-mixture fit; evaluate exclusively in Pair32 format."""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import random
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from rhpr.gr.experiment import _event, _loss, _train_generator, _write

TRIALS = [dict(id='mixed32_wide_42', seed=42, history_items=32)]
POLICY = dict(full_probability=.5, unit='optimizer_batch', inference='pair32',
              rng_stream=7301, preserve_native_positions=True)


def validate(c):
    from rhpr.gr import kindle_capacity, kindle_windows
    expected_sources = [c['parent']['kernel']]
    if c.get('resume_parent'):
        resume = c['resume_parent']
        if (resume['kernel'] != 'carasik/rhpr-kindle-mixed-views'
            or resume['git_sha'] != '192c6ce879e0135644455e4376fec2d9f7b10ae4'
            or resume['completed_epoch'] != 30):
            raise ValueError('only the saved epoch30 mixture may be resumed')
        expected_sources.append(resume['kernel'])
    if (c['kind'] != 'gr_kindle_mixed_views' or c['trials'] != TRIALS
        or c['seed'] != 42 or c['seeds'] != [42] or c['windows'] != [32]
        or c['mixed_views'] != POLICY
        or c['execution'] != dict(worker_devices=['cuda:0'], cpu_threads_per_worker=1)
        or not c['evaluation']['test_after_selection']
        or c['parent']['kernel'] != 'carasik/rhpr-kindle-long-pairs'
        or c['parent']['git_sha'] != '33754d70acc0e4cf826f139202632bf0ffe6ac5b'
        or c['kaggle']['kernel_sources'] != expected_sources
        or any(c['generator'][k] != v for k, v in kindle_capacity.WIDE.items())):
        raise ValueError('expected one seed42 native-length Full32/Pair32 mixture fit')
    normalized = copy.deepcopy(c)
    normalized.update(kind='gr_kindle_windows', trials=kindle_windows.TRIALS,
                      seeds=kindle_windows.SEEDS, windows=kindle_windows.WINDOWS)
    normalized['generator'].update(kindle_capacity.NARROW)
    normalized['execution']['worker_devices'] = ['cuda:0', 'cuda:1']
    normalized['evaluation']['test_after_selection'] = False
    kindle_windows.validate(normalized)


def locate_resume(c, input_root=Path('/kaggle/input')):
    """Cancelled kernels lack a completion manifest; verify their pinned bytes."""
    from rhpr.gr.data import file_sha256
    spec = c['resume_parent']; owner,slug = spec['kernel'].split('/')
    candidates = [input_root,input_root/slug,input_root/owner/slug,input_root/'notebooks'/owner/slug,
                  input_root/'kernels'/owner/slug]
    if input_root.exists():
        candidates.extend(p.parent for p in input_root.rglob('experiment.json') if slug in p.parts)
    for root in candidates:
        if not all((root/f).is_file() for f in spec['files']):
            continue
        for name,expected in spec['files'].items():
            path = root/name
            if path.stat().st_size != expected['size_bytes'] or file_sha256(path) != expected['sha256']:
                raise ValueError('resume artifact differs: '+name)
        old = json.loads((root/'experiment.json').read_text())
        for key in ('training','generator','data','movie_input','quantizer','evaluation','mixed_views','seed','parent'):
            if old[key] != c[key]:
                raise ValueError('resume changes '+key)
        return root
    raise FileNotFoundError('pinned cancelled mixture artifacts not mounted')


class MixedViews:
    """Independent, epoch-addressable RNG; never pads one format to the other.

    Selection is constant within an optimizer batch, including microbatches.
    Only contexts change: the caller retains the original users and targets.
    A per-epoch coverage audit rejects duplicated or missing training examples.
    """
    def __init__(self, full, pair, batch_size=512, seed=42, full_probability=.5):
        self.full = np.asarray(full)
        self.pair = np.asarray(pair)
        if (self.full.ndim != 2 or self.pair.ndim != 2 or len(full) != len(pair)
            or self.full.shape[1] != 128 or self.pair.shape[1] != 96
            or not 0 <= full_probability <= 1 or batch_size <= 0):
            raise ValueError('aligned native Full32/Pair32 contexts required')
        self.batch_size, self.seed, self.probability = batch_size, seed, full_probability
        self.epoch = None

    def schedule(self, epoch):
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, epoch, POLICY['rng_stream']]))
        return rng.random((len(self.full) + self.batch_size - 1)//self.batch_size) < self.probability

    def __call__(self, epoch, batch, part):
        if self.epoch != epoch:
            if self.epoch is not None:
                self.epoch_summary(self.epoch)
            self.epoch = epoch
            self.choices = self.schedule(epoch)
            self.counts = np.zeros(len(self.full), dtype=np.int32)
            self.examples = dict(full32=0, pair32=0)
            self.batches = dict(full32=set(), pair32=set())
        part = np.asarray(part)
        if (part.ndim != 1 or np.any(part < 0) or np.any(part >= len(self.full))
            or batch < 0 or batch >= len(self.choices)):
            raise ValueError('invalid training batch')
        np.add.at(self.counts, part, 1)
        full = bool(self.choices[batch])
        name = 'full32' if full else 'pair32'
        self.examples[name] += len(part)
        self.batches[name].add(batch)
        return (self.full if full else self.pair)[part]

    def epoch_summary(self, epoch):
        if self.epoch != epoch or not np.all(self.counts == 1):
            raise RuntimeError('mixed views must consume every original example exactly once')
        return dict(examples=dict(self.examples), batches={k:len(v) for k,v in self.batches.items()},
                    schedule_sha256=hashlib.sha256(self.choices.tobytes()).hexdigest(),
                    examples_consumed_once=True, native_sid_widths=dict(full32=128, pair32=96))


def run_trial(trial, p):
    import torch
    from rhpr.gr.data import file_sha256
    from rhpr.gr.grid import build_grid_generator
    from rhpr.gr.kindle_pairs import encode, test_selected
    from rhpr.gr.kindle_windows import encode_window
    from rhpr.gr.ml100k_pair_raw import cases
    from rhpr.gr.movie_data import case_digest
    from rhpr.gr.pair_unique4 import attach4, InputIndex4
    from rhpr.gr.sid import SidIndex
    from rhpr.gr.sid_study import state_digest

    c = copy.deepcopy(p['config'])
    c['data']['history_items'] = 32
    random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    root = Path(p['indexes'])
    idx = SidIndex.from_full_sids(np.load(root/'items-42.npy'), width=256)
    wrapped = InputIndex4(idx)
    keys = np.load(root/'pair-keys.npy'); pairs = np.load(root/'pairs-42.npy')
    model = attach4(build_grid_generator(idx.sids, c['generator'], 42, 'cpu', p['source']))
    initial = state_digest(dict(model.named_parameters()))
    baseline = json.loads((Path(p['parent'])/'trials/pair32_wide_42/trial-result.json').read_text())
    if initial != baseline['initial_state_sha256']:
        raise ValueError('mixture initializer differs from completed Pair32')
    out = Path(p['output'])/'trials'/trial['id']; out.mkdir(parents=True)
    raw = {s:cases(p['data'], s) for s in ('train', 'validation')}
    encoded = {s:(encode(h, idx.sids, keys, pairs, 32), u, t) for s,(h,u,t) in raw.items()}
    full = encode_window(raw['train'][0], idx.sids, 32)
    views = MixedViews(full, encoded['train'][0])
    model.eval()
    probe = tuple(a[:1] for a in encoded['validation'])
    with torch.inference_mode(), torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH), torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], with_flops=True) as prof:
        _loss(model, wrapped.model_batch(*probe, 'cpu'))
    forward = dict(parameters=sum(v.numel() for v in model.parameters()), encoder_positions=97,
        supported_forward_flops_batch1=sum(e.flops for e in prof.key_averages()), scope=baseline['forward']['scope'])
    if forward != baseline['forward'] or initial != state_digest(dict(model.named_parameters())):
        raise RuntimeError('mixture inference cost or initializer differs from Pair32')
    _write(out/'forward-flops.json', forward)
    provenance = dict(policy=POLICY,
        pair_case_sha256=case_digest(encoded['train']), full_case_sha256=case_digest((full,*raw['train'][1:])),
        target_count=len(raw['train'][2]), examples_doubled=False, checkpoint_resume_rng='seed/epoch-addressable')
    resume_from = None
    if p.get('resume'):
        resume_from = Path(p['resume'])/'trials'/trial['id']
        if json.loads((resume_from/'training-view-provenance.json').read_text()) != provenance:
            raise ValueError('resume changes full/pair representations or mixture RNG')
        state = torch.load(resume_from/'generator-last.pt',map_location='cpu',weights_only=False)
        if (state['epoch'] != c['resume_parent']['completed_epoch']
            or state['config']['mixed_views'] != POLICY):
            raise ValueError('resume checkpoint epoch/policy differs')
        del state
    _write(out/'training-view-provenance.json', provenance)
    model.to('cuda:0'); torch.cuda.reset_peak_memory_stats(); started = time.monotonic()
    def builder(root, ignored, split):
        if split not in encoded:
            raise ValueError('TEST forbidden during training')
        return encoded[split]
    selection = _train_generator(model, p['data'], wrapped, c, out, 'cuda:0',
                                 example_builder=builder, training_context_selector=views, resume_from=resume_from)
    epochs = selection['training_epochs']
    if (not 40 <= epochs <= 60 or selection['validation_checks'] != epochs//5
        or selection['stopping_reason'] not in ('epoch_limit','early_stopping')
        or selection['optimizer_updates'] + selection['amp_skipped_updates'] != ((len(full)+511)//512)*epochs):
        raise RuntimeError('incomplete declared mixture fit')
    curves = json.loads((out/'training-curves.json').read_text())
    timing = dict(wall_seconds=time.monotonic()-started, first_epoch_training_seconds=curves[0]['train_seconds'],
        warm_epoch_training_median_seconds=float(np.median([r['train_seconds'] for r in curves[1:]])),
        cuda_peak_allocated_bytes=int(torch.cuda.max_memory_allocated()),
        cuda_peak_reserved_bytes=int(torch.cuda.max_memory_reserved()))
    if resume_from is not None:
        timing['wall_seconds_scope'] = 'continuation session only; curve median includes both sessions'
    h,u,t = cases(p['data'], 'test')
    test = test_selected(model, (encode(h,idx.sids,keys,pairs,32),u,t), wrapped, out,
                         out/'generator-best.pt', selection['best_epoch'])
    row = dict(trial=trial['id'], seed=42, history_items=32, mode='mixed_full_pair', mixed_views=POLICY,
        initial_state_sha256=initial, sid_file_sha256=file_sha256(root/'items-42.npy'),
        pair_sid_file_sha256=file_sha256(root/'pairs-42.npy'), selection=selection, forward=forward,
        raw_case_sha256={s:case_digest(a) for s,a in raw.items()},
        case_targets_sha256={s:case_digest(a[1:]) for s,a in raw.items()},
        test_case_sha256=case_digest((h,u,t)), timing=timing, test=test, test_evaluated=True)
    for key in ('sid_file_sha256','pair_sid_file_sha256','raw_case_sha256','case_targets_sha256','test_case_sha256'):
        if row[key] != baseline[key]:
            raise RuntimeError('mixture vs completed Pair32 differs: '+key)
    _write(out/'trial-result.json', row)
    return row


def worker(plan_path, slot):
    import torch, transformers
    p = json.loads(Path(plan_path).read_text()); validate(p['config']); torch.set_num_threads(1)
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_capability() != (7,5) or transformers.__version__ != '5.22.1':
        raise RuntimeError('requires isolated T4/HF5.22.1')
    state = dict(results={}, failures=[])
    for trial in p['assignments'][slot]['trials']:
        _event('mixed_views_trial_start', trial=trial['id'])
        try:
            row = run_trial(trial,p); state['results'][trial['id']] = row
            _event('mixed_views_trial_complete', **row)
        except Exception as error:
            out = Path(p['output'])/'trials'/trial['id']; out.mkdir(parents=True,exist_ok=True)
            (out/'failure.log').write_text(traceback.format_exc())
            state['failures'].append(dict(trial=trial['id'],exception_type=type(error).__name__,message=str(error)))
            _event('mixed_views_trial_failed', **state['failures'][-1])
        _write(Path(p['output'])/'workers'/f'worker-{slot}.json', state)
        gc.collect(); torch.cuda.empty_cache()


def run(c, output):
    import torch
    from rhpr.gr.ae_suite import launch_workers
    from rhpr.gr.data import file_sha256
    from rhpr.gr.grid import prepare_grid_source
    from rhpr.gr.pair_test import locate_parent
    from rhpr.gr.resources import unpack_grid, verify_resources
    from rhpr.offline_runtime import resolve_dataset_directory

    validate(c); output=Path(output).resolve(); output.mkdir(parents=True,exist_ok=True)
    (output/'workers').mkdir(exist_ok=True); torch.set_num_threads(1)
    if torch.cuda.device_count() < 1 or torch.cuda.get_device_capability(0) != (7,5):
        raise RuntimeError('requires T4; no hardware fallback')
    parent = locate_parent(c['parent'])
    resume = locate_resume(c) if c.get('resume_parent') else None
    old = json.loads((parent/'run-manifest.json').read_text())['config']
    for key in ('training','generator','data','movie_input','quantizer','evaluation'):
        if c[key] != old[key]:
            raise ValueError('mixture comparison changes '+key)
    data = resolve_dataset_directory(c['movie_input']['path'])
    for name,sha in c['movie_input']['sha256'].items():
        if file_sha256(data/name) != sha:
            raise ValueError('input hash differs: '+name)
    indexes = output/'frozen-indexes'; indexes.mkdir()
    for name in ('items-42.npy','pairs-42.npy','pair-keys.npy'):
        shutil.copy2(parent/'frozen-indexes'/name, indexes/name)
    for rel in c['parent']['files']:
        if rel.startswith(('trials/','baselines/','control-tests/')):
            dest = output/'baselines'/rel
            dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(parent/rel,dest)
    resources = resolve_dataset_directory(c['resources']['path']); verify_resources(resources,c['resources']['manifest_sha256'])
    modern = resolve_dataset_directory(c['modern_runtime']['path']); verify_resources(modern,c['modern_runtime']['manifest_sha256'])
    scratch = Path(c['resources']['scratch_path']); scratch.mkdir(parents=True,exist_ok=True)
    source = prepare_grid_source(unpack_grid(resources/'grid-source.bin',scratch/'original'),scratch/'grid')
    overlay = scratch/'modern-site'; overlay.mkdir(parents=True,exist_ok=True)
    command = [sys.executable,'-m','pip','install','--no-index','--no-deps','--no-cache-dir','--no-compile',
        '--require-hashes','--only-binary=:all:','--target',str(overlay),'--find-links',str(modern),'-r',str(modern/'requirements-lock.txt')]
    with (output/'modern-install.log').open('w') as log:
        subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)
    plan = dict(config=c,output=str(output),source=str(source),data=str(data),indexes=str(indexes),parent=str(parent),
                assignments=[dict(slot=0,device='cuda:0',trials=TRIALS)])
    if resume is not None:
        plan['resume'] = str(resume)
    _write(output/'plan.json',plan); sys.path.insert(0,str(overlay))
    summary = launch_workers(output/'plan.json',worker_module='rhpr.gr.kindle_mixed_views',event_prefix='mixed_views')
    _write(output/'study-summary.json',dict(results=summary['results'],new_generator_fits=0 if resume else 1,
        resumed_generator_fits=1 if resume else 0,tokenizer_refitted=False,
        inference_format='pair32',training_policy=POLICY,reused_baselines=['pair32_wide_42','recent32_wide_42']))
    row = summary['results'][TRIALS[0]['id']]
    return dict(experiment=c['name'],models=1,seed=42,test_evaluated=True,
                validation_ndcg10=row['selection']['validation_ndcg10'],test_ndcg10=row['test']['metrics']['ndcg@10'])


if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--plan',required=True); parser.add_argument('--slot',type=int,required=True)
    args=parser.parse_args(); worker(args.plan,args.slot)
