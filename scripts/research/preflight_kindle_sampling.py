"""CPU audit of original Pair20 geometry and three TRAIN sampling schedules."""
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
from rhpr.gr.kindle_mixed20 import cache_plan
from rhpr.gr.kindle_pairs import encode
from rhpr.gr.kindle_random_tokenization import RandomTokenization
from rhpr.gr.kindle_sampling import ScheduledTokenization, TRIALS, probability_at, validate
from rhpr.gr.kindle_windows import encode_window
from rhpr.gr.ml100k_pair_raw import cases
from rhpr.gr.pair_unique4 import InputIndex4, attach4
from rhpr.gr.sid import SidIndex
from rhpr.gr.sid_study import state_digest


def main():
    p = argparse.ArgumentParser()
    for name in ('data', 'parent', 'cache-parent', 'source', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args(); torch.set_num_threads(1)
    c = json.loads(Path('experiments/kindle_sampling_2t4.json').read_text()); validate(c)
    pins = 0
    for label in ('parent', 'cache_parent'):
        root = getattr(a, label); spec = c[label]
        assert json.loads((root/'run-manifest.json').read_text())['git_sha'] == spec['git_sha']
        for rel, pin in spec['files'].items():
            assert (root/rel).stat().st_size == pin['size_bytes'] and file_sha256(root/rel) == pin['sha256'], rel
            pins += 1
    for rel, sha in c['movie_input']['sha256'].items(): assert file_sha256(a.data/rel) == sha
    old = json.loads((a.parent/'run-manifest.json').read_text())['config']
    for key in ('training', 'generator', 'data', 'movie_input', 'quantizer', 'evaluation'):
        assert c[key] == old[key], key
    oldkeys, sets, missing = cache_plan(a.data, a.cache_parent); assert len(missing) == 0
    root = a.cache_parent/'frozen-indexes'; sid = np.load(root/'items-42.npy'); pairs = np.load(root/'pairs-42.npy')
    h, u, t = cases(a.data, 'train'); v, vu, vt = cases(a.data, 'validation')
    previous = json.loads((a.cache_parent/'trials/mixed20_wide_42/training-curves.json').read_text())
    first_audits = {}; totals = {}; schedules = {}
    for label, spec in [(x['id'], x['sampling']) for x in TRIALS]+[('reused_p50', dict(kind='constant', probability=.5))]:
        views = ScheduledTokenization(h, sid, oldkeys, pairs, sampling=spec)
        started = time.monotonic(); order = np.random.default_rng(43).permutation(len(h))
        for batch, begin in enumerate(range(0, len(h), 512)): views(1, batch, order[begin:begin+512])
        first = views.epoch_summary(1); seconds = time.monotonic()-started
        if label == 'reused_p50':
            summary = dict(first); summary.pop('probability_schedule')
            assert summary == previous[0]['training_views']
        first_audits[label] = dict(first_epoch=first, cpu_seconds=seconds)
        counts = []; total = 0; total_forward = 0
        native = json.loads(Path('docs/gr-kindle-mixed20-preflight.json').read_text())['native_costs']
        for ep in range(1, 61):
            choices = views.schedule(ep); order = np.random.default_rng(42+ep).permutation(len(h))
            histogram = np.zeros(6, dtype=np.int64); actual = 0
            for b, begin in enumerate(range(0, len(h), 512)):
                part = order[begin:begin+512]; count = int(choices[b]); histogram[count] += len(part)
                lookup = np.r_[0, views.merge_slots(count, ep, b).cumsum()]
                actual += int(lookup[views.lengths[part]//4].sum())
            assert histogram.sum() == len(h)
            forward = sum(int(histogram[m])*native[str(m)]['supported_forward_flops_batch1'] for m in range(6))
            counts.append(dict(epoch=ep, probability=probability_at(spec, ep), merge_count_examples=histogram.tolist(),
                pair_occurrences=actual, supported_train_forward_flops=forward,
                count_schedule_sha256=hashlib.sha256(choices.tobytes()).hexdigest()))
            if label == 'reused_p50':
                assert actual == previous[ep-1]['training_views']['pair_summary_occurrences']
                assert histogram.tolist() == previous[ep-1]['training_views']['merge_count_examples']
            total += actual; total_forward += forward
        schedules[label] = counts
        totals[label] = dict(pair_occurrences=total, expected_probability_mean=float(np.mean([r['probability'] for r in counts])),
            total_examples=60*len(h), supported_forward_flops_all_train=total_forward,
            mean_train_forward_gflops_per_example=total_forward/(60*len(h))/1e9,
            note='forward-only supported matmul FLOPs; not backward/beam/elapsed time; original short-history padding retained')
    check = ScheduledTokenization(v, sid, oldkeys, pairs, sampling=TRIALS[1]['sampling'])
    for ep, expected in [(1, encode_window(v, sid, 20)), (60, encode(v, sid, oldkeys, pairs, 20))]:
        order = np.random.default_rng(42+ep).permutation(len(v)); collected = np.empty_like(expected)
        for b, begin in enumerate(range(0, len(v), 512)):
            part = order[begin:begin+512]; collected[part] = check(ep, b, part)
        np.testing.assert_array_equal(collected, expected); assert check.epoch_summary(ep)['examples_consumed_once']
    model = attach4(build_grid_generator(sid, c['generator'], 42, 'cpu', a.source))
    initial = state_digest(dict(model.named_parameters()))
    control = json.loads((a.cache_parent/'trials/mixed20_wide_42/trial-result.json').read_text())
    assert initial == control['initial_state_sha256']
    wrapped = InputIndex4(SidIndex.from_full_sids(sid, width=256)); costs = {}; losses = {}
    views = RandomTokenization(v, sid, oldkeys, pairs, history_items=20)
    for count in range(6):
        x = views.encode_batch(np.array([0, 1]), count, 1, 0)[0]; model.eval()
        with torch.inference_mode(), torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH), torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU], with_flops=True) as prof:
            loss = _loss(model, wrapped.model_batch(x[:1], vu[:1], vt[:1], 'cpu'))
        assert torch.isfinite(loss)
        costs[count] = sum(e.flops for e in prof.key_averages())
        model.train(); model.zero_grad(set_to_none=True)
        loss = _loss(model, wrapped.model_batch(x, vu[:2], vt[:2], 'cpu')); loss.backward()
        assert torch.isfinite(loss) and torch.isfinite(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.))
        losses[count] = float(loss.detach())
    assert costs[5] == 319748392 and state_digest(dict(model.named_parameters())) == initial
    result = dict(gpu_used=False, new_generator_fits=3, parent_and_cache_pins_verified=pins, input_files_verified=10,
        original_p50_first_epoch_byte_replayed=True, original_p50_all60_count_exposures_replayed=True,
        all_original_cached_sids_used_unchanged=True, new_pair_embeddings=0, tokenizer_refitted=False,
        initial_state_matches_previous_control=True, original_full_and_pair20_ramp_endpoints_verified=True,
        inference_encoder_positions=61, inference_forward_flops=319748392,
        needed_pair_counts={s:len(k) for s,k in sets.items()}, totals=totals, first_epochs=first_audits,
        native_forward_flops=costs, finite_native_forward_backward_losses=losses, all60schedules=schedules)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('parent_and_cache_pins_verified', 'totals', 'inference_forward_flops')},indent=2))


if __name__ == '__main__': main()
