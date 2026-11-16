"""CPU audit of the fixed two-cohort, three-generator-seed Recent10/20 study."""
import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import torch

from rhpr.gr.data import file_sha256
from rhpr.gr.experiment import _loss
from rhpr.gr.grid import build_grid_generator
from rhpr.gr.history_windows import encode_window, prepare_cases, trial_config, validate, verify_matched_results
from rhpr.gr.ml100k_pair_raw import cases
from rhpr.gr.movie_data import case_digest
from rhpr.gr.pair_unique4 import attach4, InputIndex4
from rhpr.gr.sid import SidIndex
from rhpr.gr.sid_study import state_digest


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--artifacts', type=Path, required=True)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); torch.set_num_threads(1)
    c = json.loads(Path('experiments/steam_recent10_20_crossed_2t4.json').read_text()); validate(c)
    stems = {'steam_a': 'steam-history', 'steam_b': 'steam-history-replica'}
    sidpath = a.artifacts/'steam-history-results/frozen-indexes/steam/items-42.npy'
    assert file_sha256(sidpath) == c['expected_sid_file_sha256']
    sid = np.load(sidpath, allow_pickle=False)
    wrapped = InputIndex4(SidIndex.from_full_sids(sid, width=256))
    reports = {}; raw_by_cohort = {}; source_users = {}
    with tempfile.TemporaryDirectory() as tmp:
        for cohort, stem in stems.items():
            data = a.artifacts/(stem+'-dataset')
            for name, digest in c['cohort_inputs'][cohort]['sha256'].items(): assert file_sha256(data/name) == digest
            _, report = prepare_cases(data, c['cohorts'][cohort], Path(tmp)/cohort)
            raw = {s: cases(Path(tmp)/cohort, s) for s in ('train', 'validation', 'test')}
            sequences = json.loads((data/'sequences.json').read_text())
            index = 0
            for row in sequences:
                seq = row['train']
                for pos in range(1, len(seq)):
                    h, u, t = (x[index] for x in raw['train'])
                    assert u == row['user_id'] and t == seq[pos] and h[h >= 0].tolist() == seq[max(0, pos-30):pos]
                    index += 1
            for split in ('validation', 'test'):
                for j, row in enumerate(sequences):
                    seq = row['train']+([row['validation']] if split == 'test' else [])
                    assert raw[split][0][j].tolist() == seq[-30:]
                    assert raw[split][1][j] == row['user_id'] and raw[split][2][j] == row[split]
            source_users[cohort] = {r['source_sasrec_user_id'] for r in json.loads((data/'user-audit.json').read_text())}
            raw_by_cohort[cohort] = raw
            reports[cohort] = dict(data=report, raw_case_sha256={s: case_digest(v) for s, v in raw.items()})
        assert len(source_users['steam_a']) == len(source_users['steam_b']) == 2000
        assert not source_users['steam_a'] & source_users['steam_b']
        for name, reference in c['reused_results'].items():
            cohort = reference['result']['dataset']; root = a.artifacts/(stems[cohort]+'-results')
            path = root/'trials'/reference['source_trial']
            assert file_sha256(path/'trial-result.json') == reference['source_result_sha256']
            assert file_sha256(path/'generator-best.pt') == reference['best_checkpoint_sha256']
            assert json.loads((root/'run-manifest.json').read_text())['git_sha'] == reference['source_sha']
            old = json.loads((root/'experiment.json').read_text())
            for key in ('training', 'generator', 'quantizer', 'expected_sid_file_sha256'):
                assert old[key] == c[key]
            row = json.loads((path/'trial-result.json').read_text()); row.update(trial=name, dataset=cohort)
            assert row == reference['result']
        checks = {}; dummy_results = {}
        for seed in c['seeds']:
            example_trial = dict(id='preflight', dataset='steam_a', seed=seed, history_items=10)
            conf = trial_config(c, example_trial)
            assert conf['seed'] == seed and conf['expected_initial_state_sha256'] == c['expected_initial_state_by_seed'][str(seed)]
            model = attach4(build_grid_generator(sid, c['generator'], seed, 'cpu', a.source))
            initial = state_digest(dict(model.named_parameters()))
            assert initial == conf['expected_initial_state_sha256']
            checks[str(seed)] = {}
            for cohort in stems:
                h, u, t = raw_by_cohort[cohort]['validation']; checks[str(seed)][cohort] = {}
                for width in c['windows']:
                    batch = wrapped.model_batch(encode_window(h[:2], sid, width), u[:2], t[:2], 'cpu')
                    model.train(); model.zero_grad(set_to_none=True)
                    loss = _loss(model, batch); loss.backward()
                    assert torch.isfinite(loss) and torch.isfinite(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.))
                    assert state_digest(dict(model.named_parameters())) == initial
                    checks[str(seed)][cohort][str(width)] = dict(loss=float(loss.detach()), finite_forward_backward=True)
            del model
        for trial in c['trials']:
            cohort = trial['dataset']; raw = raw_by_cohort[cohort]
            dummy_results[trial['id']] = dict(trial=trial['id'], dataset=cohort, seed=trial['seed'], history_items=trial['history_items'],
                initial_state_sha256=c['expected_initial_state_by_seed'][str(trial['seed'])], sid_file_sha256=file_sha256(sidpath),
                raw_case_sha256={s: case_digest(raw[s]) for s in ('train', 'validation')},
                case_targets_sha256={s: case_digest(raw[s][1:]) for s in ('train', 'validation')}, test_case_sha256=case_digest(raw['test']))
        verify_matched_results(c, dummy_results)
    result = dict(gpu_used=False, no_new_inference_of_trained_models=True, input_hashes_verified=True,
        every_prefix_and_holdout_replayed=True, original_source_user_overlap=0, cohorts=reports,
        initializer_by_seed=c['expected_initial_state_by_seed'], sid_file_sha256=file_sha256(sidpath),
        real_original_SIDs_used=True, registered_new_fits=9, verified_reused_controls=3,
        total_comparison_cells=12, finite_forward_backward=checks)
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
