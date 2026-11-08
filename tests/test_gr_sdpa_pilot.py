import copy,importlib.util,json,unittest
from pathlib import Path
from rhpr.gr.sdpa_pilot import compare_tensors,validate

ROOT=Path(__file__).resolve().parents[1]
class PilotContractTests(unittest.TestCase):
    def config(self):return json.loads((ROOT/'experiments/t5_sdpa_t4_pilot.json').read_text())
    def test_bounded_contract(self):validate(self.config())
    def test_rejects_unbounded_training(self):
        c=self.config();c['pilot']['timed_steps']=5000
        with self.assertRaises(ValueError):validate(c)
    def test_pilot_inputs_declared_private_bundle(self):
        from scripts.build_kaggle_bundle import build
        import tempfile
        with tempfile.TemporaryDirectory()as d:
            build(ROOT/'experiments/t5_sdpa_t4_pilot.json','carasik/rhpr-t5-sdpa-pilot','gpu',Path(d)/'bundle')
            m=json.loads((Path(d)/'bundle/kernel-metadata.json').read_text())
            self.assertTrue(m['is_private']);self.assertFalse(m['enable_internet'])
            self.assertIn('carasik/rhpr-t5-sdpa-runtime',m['dataset_sources'])
            self.assertEqual(m['kernel_sources'],['carasik/rhpr-ml100k-singleton-mtl'])

@unittest.skipUnless(importlib.util.find_spec('torch'),'torch required')
class ParityTests(unittest.TestCase):
    def test_accepts_small_float_errors(self):
        import torch
        r=compare_tensors({'bias':torch.ones(4)+1e-6},{'bias':torch.ones(4)},atol=1e-5,rtol=1e-5)
        self.assertLess(r['max_absolute_error'],1e-5)
    def test_rejects_wrong_gradients_and_names(self):
        import torch
        with self.assertRaises(AssertionError):compare_tensors({'q':torch.ones(4)},{'q':torch.zeros(4)},atol=1e-5,rtol=1e-5)
        with self.assertRaises(ValueError):compare_tensors({'q':torch.ones(4)},{'k':torch.ones(4)},atol=1e-5,rtol=1e-5)
    def test_rejects_nonfinite(self):
        import torch
        with self.assertRaises(FloatingPointError):compare_tensors({'q':torch.tensor(float('nan'))},{'q':torch.tensor(0.)},atol=1e-5,rtol=1e-5)
