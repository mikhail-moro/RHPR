"""CPU data/geometry/native-model checks; missing pair vectors are made on Kaggle."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from rhpr.gr.data import file_sha256
from rhpr.gr.experiment import _loss
from rhpr.gr.grid import build_grid_generator
from rhpr.gr.kindle_mixed20 import POLICY, SEEDS, cache_plan, extend_by_split, validate
from rhpr.gr.kindle_pairs import encode
from rhpr.gr.kindle_random_tokenization import RandomTokenization
from rhpr.gr.kindle_windows import encode_window
from rhpr.gr.ml100k_pair_raw import cases
from rhpr.gr.pair_unique4 import InputIndex4, attach4
from rhpr.gr.sid import SidIndex
from rhpr.gr.sid_study import state_digest


def main():
    p = argparse.ArgumentParser()
    for name in ('data', 'parent', 'source', 'output'): p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args(); torch.set_num_threads(1)
    c = json.loads(Path('experiments/kindle_mixed20_2t4.json').read_text()); validate(c)
    for filename, pin in c['parent']['files'].items():
        file = a.parent/filename
        assert file.stat().st_size == pin['size_bytes'] and file_sha256(file) == pin['sha256'], filename
    for filename, sha in c['movie_input']['sha256'].items(): assert file_sha256(a.data/filename) == sha
    old = json.loads((a.parent/'run-manifest.json').read_text())
    assert old['git_sha'] == c['parent']['git_sha']
    for key in ('training', 'generator', 'data', 'movie_input', 'quantizer', 'evaluation'):
        assert old['config'][key] == c[key], key
    indexes = a.parent/'frozen-indexes'; sid = np.load(indexes/'items-42.npy')
    oldkeys, sets, missing = cache_plan(a.data, a.parent)
    assert len(missing) == c['expected_new_pairs'] == 5011
    oldpairs = np.load(indexes/'pairs-42.npy')
    # Geometry only: not a substitute for frozen ST5+centroid prediction.
    ordinal = np.arange(len(missing)); dummy_codes = np.column_stack([ordinal % 256, ordinal//256, ordinal*0])
    keys, pairs = extend_by_split(oldkeys, oldpairs, sets, missing, dummy_codes)
    h, u, t = cases(a.data, 'train')
    views = RandomTokenization(h, sid, keys, pairs, history_items=20, augmentation_seed=42)
    started = time.monotonic(); order = np.random.default_rng(43).permutation(len(h))
    for b, begin in enumerate(range(0, len(h), 512)): views(1, b, order[begin:begin+512])
    first = views.epoch_summary(1); seconds = time.monotonic()-started
    assert first['odd_start_pair_occurrences'] > 0
    pair_count = 0; schedules = []
    for epoch in range(1, 61):
        schedule = views.schedule(epoch); order = np.random.default_rng(42+epoch).permutation(len(h))
        count = np.zeros(6, dtype=int); actual = 0
        for b, begin in enumerate(range(0, len(h), 512)):
            part = order[begin:begin+512]; m = int(schedule[b]); count[m] += len(part)
            lookup = np.r_[0, views.merge_slots(m, epoch, b).cumsum()]
            actual += int(lookup[views.lengths[part]//4].sum())
        assert int(count.sum()) == len(h)
        pair_count += actual
        schedules.append(dict(epoch=epoch, merge_count_examples=count.tolist(), pair_occurrences=actual))
    # The generalized sampler must byte-replay the original32 pilot by default.
    original = RandomTokenization(h, sid, oldkeys, oldpairs)
    for b, begin in enumerate(range(0, len(h), 512)):
        original(1, b, np.random.default_rng(43).permutation(len(h))[begin:begin+512])
    previous = json.loads(Path('docs/gr-kindle-random-tokenization-preflight.json').read_text())
    assert original.epoch_summary(1) == previous['first_epoch']
    v, vu, vt = cases(a.data, 'validation'); validation = RandomTokenization(v, sid, keys, pairs, history_items=20)
    np.testing.assert_array_equal(validation.encode_batch(np.arange(len(v)), 0, 1, 0)[0], encode_window(v, sid, 20))
    np.testing.assert_array_equal(validation.encode_batch(np.arange(len(v)), 5, 1, 0)[0], encode(v, sid, keys, pairs, 20))
    np.testing.assert_array_equal(encode_window(v, sid, 10), encode_window(v, sid, 20)[:, -40:])
    model = attach4(build_grid_generator(sid, c['generator'], 42, 'cpu', a.source))
    initial = state_digest(dict(model.named_parameters()))
    base = json.loads((a.parent/'trials/pair32_wide_42/trial-result.json').read_text())
    assert initial == base['initial_state_sha256']
    wrapped = InputIndex4(SidIndex.from_full_sids(sid, width=256)); costs = {}; losses = {}
    for label in [str(m) for m in range(6)]+['recent10']:
        x = encode_window(v[:2], sid, 10) if label == 'recent10' else validation.encode_batch(np.array([0, 1]), int(label), 1, 0)[0]
        model.eval()
        with torch.inference_mode(), torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH), torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU], with_flops=True) as prof:
            loss = _loss(model, wrapped.model_batch(x[:1], vu[:1], vt[:1], 'cpu'))
        assert torch.isfinite(loss)
        costs[label] = dict(encoder_positions=x.shape[1]+1, supported_forward_flops_batch1=sum(e.flops for e in prof.key_averages()))
        model.train(); model.zero_grad(set_to_none=True)
        loss = _loss(model, wrapped.model_batch(x, vu[:2], vt[:2], 'cpu')); loss.backward()
        assert torch.isfinite(loss) and torch.isfinite(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.))
        losses[label] = float(loss.detach())
    assert state_digest(dict(model.named_parameters())) == initial
    result = dict(parent_pins_verified=len(c['parent']['files']), input_files_verified=10, new_generator_fits=2,
        gpu_used=False, policy=POLICY, seed_roles=SEEDS, train_targets=len(h), new_pairs=len(missing),
        needed_by_split={s: len(k) for s, k in sets.items()}, original_pair_sids_unchanged=True,
        missing_pair_semantic_encoding='runtime only; CPU geometry checks use explicitly synthetic temporary input codes',
        first_epoch_geometry=first, first_epoch_cpu_seconds=seconds, sixty_epoch_pair_occurrences=pair_count,
        all_60_schedules=schedules, old32_first_epoch_byte_replay_verified=True,
        full_and_pair20_endpoints_verified=True, recent10_is_exact_latest_half=True,
        initial_state_matches_completed_controls=True, native_costs=costs, finite_forward_backward_losses=losses)
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: result[k] for k in ('new_pairs', 'needed_by_split', 'first_epoch_cpu_seconds', 'native_costs')}, indent=2))


if __name__ == '__main__': main()
