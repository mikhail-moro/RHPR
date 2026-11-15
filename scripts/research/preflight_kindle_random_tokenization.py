"""CPU real-data/protocol and native forward/backward checks before one fit."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from rhpr.gr.data import file_sha256
from rhpr.gr.experiment import _loss
from rhpr.gr.grid import build_grid_generator
from rhpr.gr.kindle_pairs import encode
from rhpr.gr.kindle_random_tokenization import POLICY, RandomTokenization, validate
from rhpr.gr.kindle_windows import encode_window
from rhpr.gr.ml100k_pair_raw import cases
from rhpr.gr.pair_unique4 import InputIndex4, attach4
from rhpr.gr.sid import SidIndex
from rhpr.gr.sid_study import state_digest


def main():
    parser = argparse.ArgumentParser()
    for name in ('data', 'parent', 'source', 'output'): parser.add_argument('--'+name, type=Path, required=True)
    a = parser.parse_args(); torch.set_num_threads(1)
    c = json.loads(Path('experiments/kindle_random_tokenization.json').read_text()); validate(c)
    for filename, pin in c['parent']['files'].items():
        path = a.parent/filename
        assert path.stat().st_size == pin['size_bytes'] and file_sha256(path) == pin['sha256'], filename
    for filename, sha in c['movie_input']['sha256'].items(): assert file_sha256(a.data/filename) == sha
    old = json.loads((a.parent/'run-manifest.json').read_text())
    assert old['git_sha'] == c['parent']['git_sha']
    for key in ('training', 'generator', 'data', 'movie_input', 'quantizer', 'evaluation'):
        assert c[key] == old['config'][key], key
    root = a.parent/'frozen-indexes'; sid = np.load(root/'items-42.npy')
    keys = np.load(root/'pair-keys.npy'); pairs = np.load(root/'pairs-42.npy')
    h, u, t = cases(a.data, 'train'); views = RandomTokenization(h, sid, keys, pairs)
    started = time.monotonic(); order = np.random.default_rng(43).permutation(len(h))
    for b, begin in enumerate(range(0, len(h), 512)): views(1, b, order[begin:begin+512])
    first = views.epoch_summary(1); preprocessing_seconds = time.monotonic()-started
    assert first['examples_consumed_once'] and first['odd_start_pair_occurrences'] > 0
    schedules = []; pair_count = 0
    for epoch in range(1, 61):
        order = np.random.default_rng(42+epoch).permutation(len(h)); schedule = views.schedule(epoch)
        counts = np.zeros(9, dtype=int); current_pairs = 0
        for b, begin in enumerate(range(0, len(h), 512)):
            part = order[begin:begin+512]; m = int(schedule[b]); counts[m] += len(part)
            lookup = np.r_[0, views.merge_slots(m, epoch, b).cumsum()]
            current_pairs += int(lookup[views.lengths[part]//4].sum())
        assert int(counts.sum()) == len(h)
        pair_count += current_pairs
        schedules.append(dict(epoch=epoch, merge_count_examples=counts.tolist(), expected_actual_pairs=current_pairs))
    vh, vu, vt = cases(a.data, 'validation'); validation = RandomTokenization(vh, sid, keys, pairs)
    np.testing.assert_array_equal(validation.encode_batch(np.arange(len(vh)), 0, 1, 0)[0], encode_window(vh, sid, 32))
    np.testing.assert_array_equal(validation.encode_batch(np.arange(len(vh)), 8, 1, 0)[0], encode(vh, sid, keys, pairs, 32))
    p1 = views.encode_batch(np.flatnonzero(views.lengths == 32)[:32], 4, 1, 0)[0]
    p2 = views.encode_batch(np.flatnonzero(views.lengths == 32)[:32], 4, 2, 0)[0]
    assert np.any(p1 != p2)
    idx = SidIndex.from_full_sids(sid, width=256); wrapped = InputIndex4(idx)
    model = attach4(build_grid_generator(sid, c['generator'], 42, 'cpu', a.source))
    initial = state_digest(dict(model.named_parameters()))
    baseline = json.loads((a.parent/'trials/pair32_wide_42/trial-result.json').read_text())
    assert initial == baseline['initial_state_sha256']
    costs = {}; losses = {}
    for m in range(9):
        x = validation.encode_batch(np.array([0, 1]), m, 1, 0)[0]
        model.eval()
        with torch.inference_mode(), torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH), torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU], with_flops=True) as prof:
            loss = _loss(model, wrapped.model_batch(x[:1], vu[:1], vt[:1], 'cpu'))
        assert torch.isfinite(loss)
        costs[str(m)] = dict(encoder_positions=x.shape[1]+1, supported_forward_flops_batch1=sum(e.flops for e in prof.key_averages()))
        model.train(); model.zero_grad(set_to_none=True)
        loss = _loss(model, wrapped.model_batch(x, vu[:2], vt[:2], 'cpu')); loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        assert torch.isfinite(loss) and torch.isfinite(norm)
        losses[str(m)] = float(loss.detach())
    assert state_digest(dict(model.named_parameters())) == initial
    assert costs['8']['supported_forward_flops_batch1'] == 516500848
    assert costs['0']['supported_forward_flops_batch1'] == 704769968
    result = dict(parent_pins_verified=len(c['parent']['files']), input_files_verified=10, gpu_used=False,
        new_generator_fits=1, policy=POLICY, train_targets=len(t), unique_train_adjacent_candidates=views.unique_adjacent_candidates,
        missing_cached_pairs=0, first_epoch=first, first_epoch_cpu_tokenization_seconds=preprocessing_seconds,
        all_60_schedules_same_original_target_count=True, sixty_epoch_pair_occurrences=pair_count,
        original_pair_sixty_epoch_occurrences=int((views.lengths//4).sum())*60,
        expected_schedule_audit=schedules, same_case_new_epoch_changes_tiling=True,
        endpoint_inputs_equal_original_full_and_pair=True, initial_state_matches_completed_pair32=True,
        native_forward_backward_losses=losses, supported_forward_costs=costs)
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: result[k] for k in ('train_targets', 'unique_train_adjacent_candidates', 'first_epoch_cpu_tokenization_seconds',
        'sixty_epoch_pair_occurrences', 'original_pair_sixty_epoch_occurrences', 'native_forward_backward_losses')}, indent=2))


if __name__ == '__main__': main()
