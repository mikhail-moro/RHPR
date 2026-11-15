"""CPU geometry/real-backbone checks; synthetic SIDs are explicitly not fitted indexes."""
import argparse
import json
import random
import tempfile
from pathlib import Path

import numpy as np
import torch

from rhpr.gr.beauty_pairs import TailTokenization, cache_sets, encode, prepare_cases, validate
from rhpr.gr.data import file_sha256
from rhpr.gr.experiment import _loss
from rhpr.gr.grid import build_grid_generator
from rhpr.gr.ml100k_pair_raw import cases
from rhpr.gr.movie_data import case_digest
from rhpr.gr.pair_unique4 import attach4, InputIndex4
from rhpr.gr.sid import SidIndex
from rhpr.gr.sid_study import state_digest


def main():
    p = argparse.ArgumentParser(); p.add_argument('--input', type=Path, required=True); p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); a = p.parse_args()
    c = json.loads(Path('experiments/beauty_pairs_2t4.json').read_text()); validate(c); torch.set_num_threads(1)
    for rel, sha in c['data']['sha256'].items(): assert file_sha256(a.input/rel) == sha
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp); (root/'resources').mkdir(); (root/'resources/data').symlink_to(a.input.resolve(), target_is_directory=True)
        _, data, report = prepare_cases(c, root/'resources', root)
        sets = cache_sets(data, 12101); assert {s: len(v) for s, v in sets.items()} == c['expected_pair_counts']
        keys = np.unique(np.concatenate(list(sets.values())))
        def synthetic(n):
            v = np.arange(n); return np.column_stack([v%256, v//256%256, v//65536, np.zeros(n)]).astype(np.int32)
        sid = synthetic(12101); pairs = synthetic(len(keys)); h, u, t = cases(data, 'train'); v, vu, vt = cases(data, 'validation')
        epochs = {}; controls = {}
        for tail in (3, 5):
            view = TailTokenization(h, sid, keys, pairs, tail=tail); order = np.random.default_rng(43).permutation(len(h))
            for b, begin in enumerate(range(0, len(h), 512)): view(1, b, order[begin:begin+512])
            epochs[str(tail)] = view.epoch_summary(1)
            np.testing.assert_array_equal(view.encode_batch(np.arange(256), view.capacity, 1, 0)[0],
                encode(h[:256], sid, keys, pairs, tail=tail))
            # Held-out caches deliberately contain only the actual fixed pairs.
            # Do not demand unused random validation/test adjacencies.
            encode(v, sid, keys, pairs, tail=tail)
        random.seed(42); np.random.seed(42); torch.manual_seed(42)
        model = attach4(build_grid_generator(sid, c['generator'], 42, 'cpu', a.source)); initial = state_digest(dict(model.named_parameters()))
        wrapped = InputIndex4(SidIndex.from_full_sids(sid, width=256))
        for width in sorted({12, 48, 52, *range(48, 81, 4)}):
            x = encode(h[:2], sid, recent=20)[:, -width:]; model.eval()
            with torch.inference_mode(), torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH), torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU], with_flops=True) as prof:
                loss = _loss(model, wrapped.model_batch(x[:1], u[:1], t[:1], 'cpu'))
            assert torch.isfinite(loss)
            model.train(); model.zero_grad(set_to_none=True)
            loss = _loss(model, wrapped.model_batch(x, u[:2], t[:2], 'cpu')); loss.backward()
            assert torch.isfinite(loss) and torch.isfinite(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.))
            controls[str(width)] = dict(encoder_positions=width+1, supported_forward_flops_batch1=sum(e.flops for e in prof.key_averages()))
        assert initial == state_digest(dict(model.named_parameters()))
        result = dict(gpu_used=False, warning='Synthetic unique3+1SIDs ONLY for geometry/backbone,not real fitted indexes or quality claims',
            data_sha256_verified=c['data']['sha256'], data=report, pair_counts={s:len(v) for s,v in sets.items()}, all_pairs=len(keys),
            expected_initial_state_sha256=initial, parameters=sum(v.numel() for v in model.parameters()),
            raw_case_sha256={s:case_digest(cases(data,s)) for s in sets}, first_epochs_synthetic=epochs, native_costs=controls)
        a.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
        print(json.dumps({k: result[k] for k in ('all_pairs', 'parameters', 'native_costs')}, indent=2))


if __name__ == '__main__': main()
