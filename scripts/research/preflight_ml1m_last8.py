"""CPU-only causal-data and ordinary TIGER shape/gradient/beam preflight."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from rhpr.gr.experiment import _loss, _model_batch
from rhpr.gr.grid import build_grid_generator
from rhpr.gr.ml100k_pair_raw import cases
from rhpr.gr.ml1m_last8 import WINDOWS, prepare_raw, validate
from rhpr.gr.sid_study import state_digest


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--archive', type=Path, default=Path('/tmp/ml-1m.zip'))
    p.add_argument('--data', type=Path, default=Path('/tmp/ml1m-last8-prepared'))
    p.add_argument('--source', type=Path, default=Path('/workspace/artifacts/history-length-input/resources/grid-prepared'))
    p.add_argument('--output', type=Path, default=Path('docs/gr-ml1m-last8-preflight.json'))
    a = p.parse_args(); torch.set_num_threads(1)
    c = json.loads(Path('experiments/ml1m_last8_windows_2t4.json').read_text()); validate(c)
    _, report = prepare_raw(a.archive, a.data, c['data'])
    assert {s:v['raw_case_sha256'] for s,v in report['splits'].items()} == c['expected_raw_case_sha256']
    raw = {s:cases(a.data, s) for s in ('train', 'validation', 'test')}
    # Constructor tensors need unique paths; actual learned SIDs are fit once on
    # Kaggle. These fixtures test shapes/gradients, not tokenizer quality/parity.
    sid = np.array([[i%256, (i//256)%256, 0, 0] for i in range(3706)], dtype=np.int64)
    model = build_grid_generator(sid, c['generator'], 42, 'cpu', a.source)
    initial = state_digest(dict(model.named_parameters()))
    assert initial == c['expected_initial_state_sha256']
    assert not hasattr(model, 'pair_input_embedding')
    costs = {}
    # Select genuinely long users for the CPU probes, not predominantly padding.
    h,u,t = raw['train']; rows = np.argsort((h>=0).sum(1))[-2:]
    for w in WINDOWS:
        batch = _model_batch(h[rows, -w:], u[rows], t[rows], sid, 'cpu')
        model.eval(); one = {k:v[:1] for k,v in batch.items()}
        with torch.inference_mode(), torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH), torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU], with_flops=True) as prof:
            loss = _loss(model, one)
        assert torch.isfinite(loss)
        model.train(); model.zero_grad(set_to_none=True)
        loss = _loss(model, batch); loss.backward()
        assert torch.isfinite(loss) and torch.isfinite(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.))
        assert state_digest(dict(model.named_parameters())) == initial
        costs[str(w)] = dict(encoder_positions=4*w+1, forward_flops=sum(e.flops for e in prof.key_averages()),
            finite_forward_backward=True, physical_batch=c['physical_batch_by_window'][str(w)],
            evaluation_batch=c['evaluation_batch_by_window'][str(w)])
    model.eval()
    with torch.inference_mode():
        batch = _model_batch(h[rows[:1], -8:], u[rows[:1]], t[rows[:1]], sid, 'cpu')
        paths, scores = model.generate(attention_mask=batch['attention_mask_encoder'], input_ids=batch['input_ids'], user_id=batch['user_id'])
    assert paths.shape == (1,20,4) and torch.isfinite(scores).all()
    result = dict(gpu_used=False, source_zip_sha256=report['source_zip_sha256'], data_counts={k:report[k] for k in ('users','items','events','train_examples','train_events')},
        raw_case_sha256=c['expected_raw_case_sha256'], targets_shared=True, synthetic_unique_sids=True,
        fitted_index_tested=False, initial_state_sha256=initial, parameters=sum(v.numel() for v in model.parameters()),
        native_encoder=True, finite_beam20=True, costs=costs)
    a.output.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result))


if __name__ == '__main__': main()
