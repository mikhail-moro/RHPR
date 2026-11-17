"""CPU audit of the one-fit Beauty Recent12 control using saved actual SIDs."""
import json
import argparse
from pathlib import Path
import tempfile

import numpy as np
import torch

from rhpr.gr.beauty_pairs import encode, prepare_cases, validate
from rhpr.gr.data import file_sha256
from rhpr.gr.experiment import _loss
from rhpr.gr.grid import build_grid_generator
from rhpr.gr.ml100k_pair_raw import cases
from rhpr.gr.movie_data import case_digest
from rhpr.gr.pair_unique4 import InputIndex4, attach4
from rhpr.gr.sid import SidIndex
from rhpr.gr.sid_study import state_digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=Path('experiments/beauty_recent12_control.json'))
    parser.add_argument('--output', type=Path, default=Path('docs/gr-beauty-recent12-preflight.json'))
    args = parser.parse_args()
    c = json.loads(args.config.read_text()); validate(c)
    windows = [t['recent'] for t in c['trials']]
    old = Path('/workspace/artifacts/beauty-pair/results')
    data = Path('/workspace/artifacts/beauty-pair/input')
    source = Path('/workspace/artifacts/history-length-input/resources/grid-prepared')
    torch.set_num_threads(1)
    for name, digest in c['data']['sha256'].items(): assert file_sha256(data/name) == digest
    sid_path = old/'frozen-indexes/items-42.npy'
    assert file_sha256(sid_path) == c['expected_sid_file_sha256']
    sid = np.load(sid_path); ref = c['reference_result']
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp); (root/'resources').mkdir(); (root/'resources/data').symlink_to(data)
        _, prepared, report = prepare_cases(c, root/'resources', root)
        raw = {s: cases(prepared, s) for s in ('train', 'validation', 'test')}
        assert {s: case_digest(raw[s]) for s in ('train', 'validation')} == ref['raw_case_sha256']
        assert {s: case_digest(raw[s][1:]) for s in ('train', 'validation')} == ref['case_targets_sha256']
        assert case_digest(raw['test']) == ref['test_case_sha256']
        for h, u, t in raw.values():
            for width in windows:
                x = encode(h, sid, recent=width)
                assert x.shape == (len(t), 4*width)
                np.testing.assert_array_equal(x, encode(h, sid, recent=20)[:, -4*width:])
        model = attach4(build_grid_generator(sid, c['generator'], 42, 'cpu', source))
        initial = state_digest(dict(model.named_parameters())); assert initial == c['expected_initial_state_sha256']
        wrapped = InputIndex4(SidIndex.from_full_sids(sid, width=256))
        h, u, t = raw['train']; costs = {}
        for width in windows:
            x = encode(h[:2], sid, recent=width); model.eval()
            with torch.inference_mode(), torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH), torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU], with_flops=True) as prof:
                loss = _loss(model, wrapped.model_batch(x[:1], u[:1], t[:1], 'cpu'))
            assert torch.isfinite(loss)
            model.train(); model.zero_grad(set_to_none=True)
            loss = _loss(model, wrapped.model_batch(x, u[:2], t[:2], 'cpu')); loss.backward()
            assert torch.isfinite(loss) and torch.isfinite(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.))
            assert state_digest(dict(model.named_parameters())) == initial
            costs[str(width)] = dict(encoder_positions=4*width+1,
                forward_flops=sum(e.flops for e in prof.key_averages()), finite_forward_backward=True)
        result = dict(gpu_used=False, new_generator_fits=len(c['trials']), fitted_sids_used=True,
            local_saved_index_only=True, external_vector_transfer=False, data=report,
            data_sha256=c['data']['sha256'], raw_cases_verified=True, targets_verified=True,
            sid_file_sha256=file_sha256(sid_path), initial_state_sha256=initial,
            parameters=sum(v.numel() for v in model.parameters()), costs=costs)
        if len(windows) == 1: result.update(costs[str(windows[0])])
        args.output.write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps({k: result[k] for k in ('parameters', 'initial_state_sha256', 'costs')}))


if __name__ == '__main__': main()
