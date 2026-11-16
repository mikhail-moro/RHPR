"""Verify frozen cases and real TIGER shapes on CPU; synthetic SIDs are not quality evidence."""
import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import torch

from rhpr.gr.data import file_sha256
from rhpr.gr.experiment import _loss
from rhpr.gr.grid import build_grid_generator
from rhpr.gr.history_windows import encode_window, prepare_cases, validate
from rhpr.gr.ml100k_pair_raw import cases
from rhpr.gr.movie_data import case_digest
from rhpr.gr.pair_unique4 import attach4, InputIndex4
from rhpr.gr.sid import SidIndex
from rhpr.gr.sid_study import state_digest


def main():
    p = argparse.ArgumentParser(); p.add_argument('--input', type=Path, required=True)
    p.add_argument('--source', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); torch.set_num_threads(1)
    c = json.loads(Path('experiments/steam_history_windows_2t4.json').read_text()); validate(c)
    for name, digest in c['pilot_input']['sha256'].items(): assert file_sha256(a.input/name) == digest
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp); _, report = prepare_cases(a.input, c['cohorts']['steam'], root/'cases')
        raw = {s: cases(root/'cases', s) for s in ('train', 'validation', 'test')}
        original = json.loads((a.input/'sequences.json').read_text())
        assert len(original) == 2000 and len(raw['train'][2]) == 127023
        # Independently verify the source-prefix schedule and both temporal holdouts.
        at = 0
        for row in original:
            seq = row['train']; uid = row['user_id']
            for target in range(1, len(seq)):
                h, u, t = (v[at] for v in raw['train'])
                assert u == uid and t == seq[target] and h[h >= 0].tolist() == seq[max(0, target-30):target]
                at += 1
        for split in ('validation', 'test'):
            h, u, t = raw[split]
            for j, row in enumerate(original):
                seq = row['train'] + ([row['validation']] if split == 'test' else [])
                assert u[j] == row['user_id'] and t[j] == row[split] and h[j].tolist() == seq[-30:]
        n = c['cohorts']['steam']['items']; v = np.arange(n)
        sid = np.column_stack([v % 256, v//256 % 256, v//65536, np.zeros(n)]).astype(np.int32)
        wrapped = InputIndex4(SidIndex.from_full_sids(sid, width=256))
        model = attach4(build_grid_generator(sid, c['generator'], 42, 'cpu', a.source))
        initial = state_digest(dict(model.named_parameters())); assert initial == c['expected_initial_state_sha256']
        costs = {}
        for width in (10, 20, 30):
            h, u, t = raw['validation']; encoded = encode_window(h[:2], sid, width)
            model.eval()
            with torch.inference_mode(), torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH), torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU], with_flops=True) as prof:
                loss = _loss(model, wrapped.model_batch(encoded[:1], u[:1], t[:1], 'cpu'))
            assert torch.isfinite(loss)
            model.train(); model.zero_grad(set_to_none=True)
            loss = _loss(model, wrapped.model_batch(encoded, u[:2], t[:2], 'cpu')); loss.backward()
            assert torch.isfinite(loss) and torch.isfinite(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.))
            costs[str(width)] = dict(encoder_positions=4*width+1,
                supported_forward_flops_batch1=sum(e.flops for e in prof.key_averages()))
        assert initial == state_digest(dict(model.named_parameters()))
        result = dict(gpu_used=False, fitted_index=False, synthetic_SIDs_for_shape_only=True, data=report,
            source_prefix_schedule_independently_verified=True, raw_case_sha256={s: case_digest(v) for s, v in raw.items()},
            parameters=sum(p.numel() for p in model.parameters()), initial_state_sha256=initial, costs=costs,
            input_sha256=c['pilot_input']['sha256'])
        a.output.write_text(json.dumps(result, indent=2)+'\n'); print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
