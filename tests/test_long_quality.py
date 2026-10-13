from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    import numpy as np
    import torch
except ImportError:
    np = torch = None


@unittest.skipIf(torch is None, "Torch and numpy required")
class LongQualityTest(unittest.TestCase):
    def config(self):
        return json.loads((ROOT / "experiments/yambda_hstu_long_compression_quality.json").read_text())

    def small_config(self):
        config = self.config()
        config["model"].update(d_model=8, n_layers=1, n_heads=1, dropout=0.)
        config["training"].update(raw_context_length=64, raw_tail_size=8, supervised_targets=4,
                                  num_negatives=7, effective_batch_size=4, amp=False)
        return config

    def cohort(self):
        from rhpr.sasrec.long_history_data import LongHistoryCohort, LongHistoryUser
        users = tuple(LongHistoryUser(i, (np.arange(90 + i) % 11 + 1).astype(np.int32),
                                      np.arange(90 + i, dtype=np.uint32)) for i in range(5))
        return LongHistoryCohort(users, np.arange(11, dtype=np.int64), {})

    def test_registered_configuration_and_dispatch(self):
        from rhpr.sasrec.long_quality import validate_long_quality_config, quality_variants
        from rhpr.experiment import run_experiment
        config = self.config()
        validate_long_quality_config(config)
        self.assertEqual(len(quality_variants(config["summary_budgets"])), 13)
        for path, value in [(('training', 'supervised_targets'), 128),
                            (('training', 'physical_batch_sizes'), [3]),
                            (('training', 'weight_decay'), float('nan')),
                            (('training', 'budget_safety_factor'), .5),
                            (('summary_budgets',), [32, 16, 8])]:
            bad = copy.deepcopy(config)
            cursor = bad
            for key in path[:-1]:
                cursor = cursor[key]
            cursor[path[-1]] = value
            with self.assertRaises(ValueError):
                validate_long_quality_config(bad)
        with self.assertRaisesRegex(ValueError, "output_dir"):
            run_experiment(config)

    def test_backbone_initialization_is_identical_across_all_thirteen_variants(self):
        from rhpr.sasrec.long_quality import build_quality_model, quality_variants
        config = self.small_config()
        baseline = None
        for variant in quality_variants(config["summary_budgets"]):
            model = build_quality_model(config, variant, 11)
            state = {k: v for k, v in model.state_dict().items() if not k.startswith('compressor.')}
            if baseline is None:
                baseline = state
            self.assertEqual(state.keys(), baseline.keys())
            self.assertTrue(all(torch.equal(state[k], baseline[k]) for k in state))
            self.assertEqual(model.hstu_normalization_length, 27)

    def test_all_variants_complete_matched_training_and_serve_one_final_state(self):
        from rhpr.sasrec.long_quality import build_quality_model, quality_variants, train_quality_pass, _optimizer, _scaler
        from rhpr.sasrec.long_quality_schedule import quality_window_schedule
        config, cohort = self.small_config(), self.cohort()
        windows = quality_window_schedule(cohort, seed=42, epoch=1, raw_context_length=64)
        device = torch.device('cpu')
        torch.set_num_threads(1)
        for variant in quality_variants(config["summary_budgets"]):
            model = build_quality_model(config, variant, 11)
            optimizer = _optimizer(model, config["training"])
            result = train_quality_pass(model, cohort, windows, config=config, variant=variant,
                         physical_batch=2, device=device, optimizer=optimizer, scaler=_scaler(device, config["training"]))
            self.assertEqual(result['windows'], 5)
            self.assertEqual(result['labels'], 20)
            self.assertEqual(result['successful_updates'], 2)
            self.assertEqual(result['amp_skipped_update_attempts'], 0)
            self.assertTrue(np.isfinite(result['loss']))
            self.assertTrue(torch.equal(model.item_embedding.weight[0], torch.zeros(8)))
            model.compressor.tail_size = 8
            model.eval()
            ids = torch.tensor(cohort.users[0].train[-(8 if variant['kind'] == 'recent' else 64):].copy())[None, :].long()
            state = model.last_state(ids, torch.ones_like(ids, dtype=torch.bool))
            self.assertEqual(state.shape, (1, 8))
            self.assertTrue(torch.isfinite(state).all())

    def test_partial_effective_batch_gradients_match_with_different_physical_batch_sizes(self):
        from rhpr.sasrec.long_quality import build_quality_model, train_quality_pass, _optimizer, _scaler
        from rhpr.sasrec.long_quality_schedule import quality_window_schedule
        config, cohort = self.small_config(), self.cohort()
        windows = quality_window_schedule(cohort, seed=42, epoch=2, raw_context_length=64)
        variant = dict(kind='pair_mlp', summary_tokens=4, name='pair_mlp_m4')
        models = []
        device = torch.device('cpu')
        for physical in (2, 4):
            model = build_quality_model(config, variant, 11)
            train_quality_pass(model, cohort, windows, config=config, variant=variant, physical_batch=physical,
                 device=device, optimizer=_optimizer(model, config['training']), scaler=_scaler(device, config['training']))
            models.append(model)
        for name, value in models[0].state_dict().items():
            torch.testing.assert_close(value, models[1].state_dict()[name], atol=2e-6, rtol=2e-5)

    def test_nonfinite_aggregate_norm_retries_without_mutating_optimizer(self):
        from rhpr.sasrec.long_quality import build_quality_model, train_quality_pass, _optimizer
        from rhpr.sasrec.long_quality_schedule import quality_window_schedule
        config, cohort = self.small_config(), self.cohort()
        windows = quality_window_schedule(cohort, seed=42, epoch=3, raw_context_length=64)
        variant = dict(kind='recent', summary_tokens=0, name='recent128')
        model = build_quality_model(config, variant, 11)
        optimizer = _optimizer(model, config['training'])
        original = torch.nn.utils.clip_grad_norm_
        calls = 0
        class RecoveryScaler:
            def __init__(self):
                self.steps, self.value, self.reductions = 0, 1024., 0
            def scale(self, value): return value
            def unscale_(self, optimizer): pass
            def is_enabled(self): return True
            def get_scale(self): return self.value
            def update(self, new_scale=None):
                if new_scale is not None:
                    self.value = new_scale
                    self.reductions += 1
            def step(self, optimizer):
                self.steps += 1
                self.assert_optimizer_safe()
                optimizer.step()
            def assert_optimizer_safe(self):
                if calls == 1:
                    raise AssertionError('optimizer mutated after nonfinite norm')
        scaler = RecoveryScaler()
        def injected_norm(parameters, max_norm):
            nonlocal calls
            calls += 1
            return torch.tensor(float('inf')) if calls == 1 else original(parameters, max_norm)
        with patch('torch.nn.utils.clip_grad_norm_', side_effect=injected_norm):
            result = train_quality_pass(model, cohort, windows, config=config, variant=variant,
                 physical_batch=2, device=torch.device('cpu'), optimizer=optimizer, scaler=scaler)
        self.assertEqual(result['successful_updates'], 2)
        self.assertEqual(result['amp_skipped_update_attempts'], 1)
        self.assertEqual(scaler.steps, 2)
        self.assertEqual(scaler.reductions, 1)

    def test_complete_small_pipeline_selects_validation_and_emits_exact_catalogue_results(self):
        from rhpr.sasrec.long_quality import run_long_quality
        config, cohort = self.small_config(), self.cohort()
        config['training'].update(min_passes=1, max_passes=2, validation_interval=1)
        config['benchmark'].update(warmup=0, iterations=1, batch_sizes=[1, 2], evaluation_batch_size=2)
        cohort.summary.update({k: config['data'][k] for k in (
            'train_end_timestamp', 'cohort_uid_length_sha256', 'catalog_sha256', 'cohort_mapped_train_items_sha256')})
        evaluation = SimpleNamespace(summary={
            'evaluation_content_sha256': config['data']['evaluation_content_sha256'],
            'validation_users': 5, 'test_users': 5,
        })
        def examples(cohort, evaluation, *, split_name, max_seq_len):
            for user in cohort.users:
                yield SimpleNamespace(user_id=user.user_id, context=user.train[-max_seq_len:],
                                      target=0 if user.user_id == 0 else int(user.train[-1] % 11 + 1))
        def preflight(config, variant, cohort, evaluation, schedule, device):
            return {**variant, 'training': [], 'inference': [
                dict(status='ok', batch_size=2, users_per_second=1000.)],
                'selected_physical_batch': 2, 'estimated_training_pass_seconds': .1}
        module = 'rhpr.sasrec.long_quality'
        with tempfile.TemporaryDirectory() as temp, \
             patch(module + '.validate_long_quality_config'), \
             patch(module + '.resolve_dataset_directory', side_effect=Path), \
             patch(module + '.load_prepared_cohort', return_value=cohort), \
             patch(module + '.load_prepared_long_evaluation', return_value=evaluation), \
             patch(module + '.iter_long_evaluation_examples', side_effect=examples), \
             patch(module + '._preflight_variant', side_effect=preflight), \
             patch(module + '._cleanup'), \
             patch(module + '._registered_device', return_value=torch.device('cpu')), \
             patch(module + '.torch.cuda.manual_seed_all'), \
             patch(module + '.torch.cuda.get_device_name', return_value='cpu-contract-test'), \
             patch('builtins.print'):
            result = run_long_quality(config, Path(temp))
            report = json.loads((Path(temp) / 'long-quality.json').read_text())
            self.assertTrue(result['quality_complete'])
            self.assertTrue(report['complete'])
            self.assertEqual(len(report['models']), 13)
            self.assertEqual(result['labels_per_model'], 40)
            self.assertEqual(len(list(Path(temp).glob('*-test-users.json'))), 13)
            self.assertFalse(list(Path(temp).rglob('*.pt')))
            for record in report['models'].values():
                self.assertIn(record['best_epoch'], (1, 2))
                self.assertEqual(len(record['passes']), 2)
                self.assertEqual(record['test']['anchor_count'], 5)
                self.assertEqual(record['test']['oov_target_count'], 1)
                self.assertEqual(record['test']['num_items'], 11)
                self.assertEqual(record['trained_inference']['metadata']['retained_tail_tokens'], 8)


if __name__ == '__main__':
    unittest.main()
