"""CPU provenance, layout, FLOP and native frozen-checkpoint replay checks."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from rhpr.gr.data import file_sha256
from rhpr.gr.experiment import _evaluate, _loss
from rhpr.gr.grid import build_grid_generator
from rhpr.gr.kindle_alignment import VARIANTS, encode_layout, original_prediction, pair_keys, validate
from rhpr.gr.kindle_pairs import encode
from rhpr.gr.kindle_windows import encode_window
from rhpr.gr.ml100k_pair_raw import cases
from rhpr.gr.pair_unique4 import InputIndex4, attach4
from rhpr.gr.sid import SidIndex
from rhpr.gr.sid_study import state_digest


def main():
    parser = argparse.ArgumentParser()
    for name in ('data', 'pair', 'mixed', 'source', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    a = parser.parse_args(); torch.set_num_threads(1)
    c = json.loads(Path('experiments/kindle_alignment_2t4.json').read_text()); validate(c)
    parents = dict(pair=a.pair, mixed=a.mixed); verified = 0
    for name, spec in c['parents'].items():
        for filename, pin in spec['files'].items():
            path = parents[name]/filename
            assert path.stat().st_size == pin['size_bytes'] and file_sha256(path) == pin['sha256'], filename
            verified += 1
        manifest = json.loads((parents[name]/'run-manifest.json').read_text())
        assert manifest['git_sha'] == spec['git_sha']
        for key in ('generator', 'data', 'movie_input'):
            assert c[key] == manifest['config'][key], key
    for filename, sha in c['movie_input']['sha256'].items():
        assert file_sha256(a.data/filename) == sha
    root = a.pair/'frozen-indexes'; sid = np.load(root/'items-42.npy')
    idx = SidIndex.from_full_sids(sid, width=256); wrapped = InputIndex4(idx)
    keys = np.load(root/'pair-keys.npy'); sids = np.load(root/'pairs-42.npy')
    needed = []; coverage = {}; raw = {s: cases(a.data, s) for s in c['evaluation']['splits']}
    for split, (h, u, t) in raw.items():
        assert len(t) == 958 and np.all(h[:, -34:] >= 0)
        np.testing.assert_array_equal(encode_layout(h, '32_reference', sid, keys, sids), encode(h, sid, keys, sids, 32))
        np.testing.assert_array_equal(encode_layout(h, '32_single', sid, keys, sids), encode_window(h, sid, 32))
        for v in VARIANTS:
            k = pair_keys(h, v, len(sid)); needed.append(k)
            coverage[split+'_'+v] = dict(unique_pairs=len(k), missing_pairs=len(np.setdiff1d(k, keys)))
    missing = np.setdiff1d(np.unique(np.concatenate(needed)), keys)
    assert len(missing) == c['expected_new_pairs']
    costs = {}; replay = {}; h, u, t = raw['validation']
    for name in ('mixed32', 'pair32'):
        parent = a.mixed if name == 'mixed32' else a.pair
        ck = torch.load(parent/'trials'/(name+'_wide_42')/'generator-best.pt', map_location='cpu', weights_only=False)
        assert ck['epoch'] == 55 and ck['config']['seed'] == 42 and ck['config']['generator'] == c['generator']
        model = attach4(build_grid_generator(sid, c['generator'], 42, 'cpu', a.source))
        model.load_state_dict(ck['model'], strict=True); del ck
        model.requires_grad_(False); model.eval(); before = state_digest(model.state_dict()); costs[name] = {}
        for v in VARIANTS:
            x = encode_layout(h[:1], v, sid, keys, sids)
            with torch.inference_mode(), torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH), torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU], with_flops=True) as prof:
                loss = _loss(model, wrapped.model_batch(x, u[:1], t[:1], 'cpu'))
            assert torch.isfinite(loss)
            costs[name][v] = dict(encoder_positions=x.shape[1]+1,
                supported_forward_flops_batch1=sum(e.flops for e in prof.key_averages()), probe_loss=float(loss))
        assert costs[name]['32_reference']['supported_forward_flops_batch1'] == 516500848
        assert costs[name]['32_single']['supported_forward_flops_batch1'] == 704769968
        x = encode_layout(h[:8], '32_reference', sid, keys, sids)
        _, pred, scores = _evaluate(model, (x, u[:8], t[:8]), wrapped, 64, 'cpu', 'fp32')
        with np.load(original_prediction(parents, name, 'validation')) as z:
            np.testing.assert_array_equal(pred, z['semantic_ids'][:8])
            score_delta = float(np.max(np.abs(scores-z['scores'][:8])))
        assert state_digest(model.state_dict()) == before and all(not p.requires_grad and p.grad is None for p in model.parameters())
        replay[name] = dict(eight_reference_rankings_identical=True, max_abs_score_difference=score_delta,
            loaded_state_sha256=before, model_unchanged=True)
    result = dict(parent_pins_verified=verified, input_files_verified=10, training_performed=False, gpu_used=False,
        all_958_users_per_split_have_34_past_events=True, coverage=coverage, new_pairs=len(missing),
        reference_and_single_encoders_equal_parents=True, costs=costs, native_checkpoint_replay=replay)
    a.output.write_text(json.dumps(result, indent=2)+'\n'); print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
