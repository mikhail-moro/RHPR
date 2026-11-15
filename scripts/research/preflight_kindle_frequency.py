"""Verify frozen parents, exact counts, all native shapes and initializer on CPU."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from rhpr.gr.data import file_sha256
from rhpr.gr.experiment import _loss
from rhpr.gr.grid import build_grid_generator
from rhpr.gr.kindle_frequency import FrequencyTokenization, cache_plan, train_support, validate, TRIALS
from rhpr.gr.kindle_mixed20 import extend_by_split
from rhpr.gr.ml100k_pair_raw import cases
from rhpr.gr.pair_unique4 import InputIndex4, attach4
from rhpr.gr.sid import SidIndex
from rhpr.gr.sid_study import state_digest


def main():
    p = argparse.ArgumentParser()
    for name in ('data', 'parent', 'cache-parent', 'source', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args(); torch.set_num_threads(1)
    c = json.loads(Path('experiments/kindle_frequency_2t4.json').read_text()); validate(c)
    for label in ('parent', 'cache_parent'):
        root = getattr(a, label)
        assert json.loads((root/'run-manifest.json').read_text())['git_sha'] == c[label]['git_sha']
        for rel, pin in c[label]['files'].items():
            assert (root/rel).stat().st_size == pin['size_bytes'] and file_sha256(root/rel) == pin['sha256'], rel
    for rel, sha in c['movie_input']['sha256'].items(): assert file_sha256(a.data/rel) == sha
    old = json.loads((a.parent/'run-manifest.json').read_text())['config']
    for key in ('training', 'generator', 'data', 'movie_input', 'quantizer', 'evaluation'):
        assert c[key] == old[key], key
    support = train_support(a.data, c['data']['items'])
    assert hashlib.sha256(support.astype('<i4').tobytes()).hexdigest() == c['train_support_sha256']
    oldkeys, sets, missing = cache_plan(a.data, a.cache_parent)
    assert len(missing) == c['expected_new_pairs'] == 5055
    indexes = a.cache_parent/'frozen-indexes'; sid = np.load(indexes/'items-42.npy')
    ordinal = np.arange(len(missing)); dummy = np.column_stack([ordinal % 256, ordinal//256, ordinal*0])
    keys, pairs = extend_by_split(oldkeys, np.load(indexes/'pairs-42.npy'), sets, missing, dummy)
    h, u, t = cases(a.data, 'train'); v, vu, vt = cases(a.data, 'validation')
    audits = {}; shapes = {}; schedules = None
    for trial in TRIALS:
        mode = trial['mode']; budget = trial['max_pairs']; label = trial['id']
        started = time.monotonic(); views = FrequencyTokenization(h, sid, keys, pairs, support, mode, max_pairs=budget)
        order = np.random.default_rng(43).permutation(len(h))
        for batch, begin in enumerate(range(0, len(h), 512)):
            views(1, batch, order[begin:begin+512])
        audits[label] = dict(first_epoch=views.epoch_summary(1), first_epoch_cpu_seconds=time.monotonic()-started)
        current = [hashlib.sha256(views.schedule(ep).tobytes()).hexdigest() for ep in range(1, 61)]
        if schedules is None: schedules = current
        elif budget == 5: assert current == schedules
        vv = FrequencyTokenization(v, sid, keys, pairs, support, mode, max_pairs=budget)
        encoded = vv.evaluation(); assert encoded.shape == (958, 4*(20-budget)) and np.all(encoded >= 0)
        shapes[label] = [vv.encode_batch(np.array([0, 1]), m, 1, 0)[0] for m in range(budget+1)]
        vv.evaluation(); audits[label]['validation_rare_items_merged'] = int(vv.last_rare_per_row.sum())
        audits[label]['validation_all_rare_preserved_fraction'] = float(np.mean(vv.last_rare_per_row == 0))
    aa, bb = [audits[m]['first_epoch'] for m in ('frequency20_budget5_42', 'random20_budget5_42')]
    for key in ('pair_summary_occurrences', 'padding_sid_positions', 'schedule_sha256', 'merge_count_batches', 'merge_count_examples'):
        assert aa[key] == bb[key], key
    assert aa['rare_context_items_merged'] < bb['rare_context_items_merged']
    model = attach4(build_grid_generator(sid, c['generator'], 42, 'cpu', a.source))
    initial = state_digest(dict(model.named_parameters()))
    baseline = json.loads((a.parent/'trials/pair32_wide_42/trial-result.json').read_text())
    assert initial == baseline['initial_state_sha256']
    wrapped = InputIndex4(SidIndex.from_full_sids(sid, width=256)); costs = {}; losses = {}
    for mode in shapes:
        for m, x in enumerate(shapes[mode]):
            model.eval()
            with torch.inference_mode(), torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH), torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU], with_flops=True) as prof:
                loss = _loss(model, wrapped.model_batch(x[:1], vu[:1], vt[:1], 'cpu'))
            flops = sum(e.flops for e in prof.key_averages())
            if m in costs: assert costs[m] == flops
            costs[m] = flops
            model.train(); model.zero_grad(set_to_none=True)
            loss = _loss(model, wrapped.model_batch(x, vu[:2], vt[:2], 'cpu')); loss.backward()
            assert torch.isfinite(loss) and torch.isfinite(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.))
            losses[f'{mode}_{m}'] = float(loss.detach())
    assert costs[5] == 319748392 and costs[0] == 427087952
    assert state_digest(dict(model.named_parameters())) == initial
    report = dict(gpu_used=False, new_generator_fits=3, original_and_cache_pins_verified=69, input_pins_verified=10,
        initializer_matches_completed_controls=True, target_count=len(t),
        missing_pair_geometry='synthetic temporary codes ONLY; actual frozen ST5 vectors and assignments computed on Kaggle',
        missing_pairs=len(missing), needed_by_split={s: len(k) for s, k in sets.items()},
        original_pair_sids_preserved=True, all_60_budget5_schedules_match=True,
        native_flops_by_pair_count=costs, finite_native_forward_backward=losses, audits=audits)
    a.output.write_text(json.dumps(report, indent=2)+'\n'); print(json.dumps(report, indent=2))


if __name__ == '__main__': main()
