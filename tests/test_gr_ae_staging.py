"""Warm-up rate estimator, phase transfer and actual two-worker freeze runs."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = os.environ.get("RHPR_GRID_SOURCE")
NUMPY = importlib.util.find_spec("numpy")
READY = SOURCE and all(importlib.util.find_spec(n) for n in ("torch", "numpy", "hydra", "lightning", "transformers"))


@unittest.skipUnless(NUMPY, "Requires NumPy")
class SpectrumTests(unittest.TestCase):
    def test_water_level_matches_analytic_rates_and_pca_is_rotation_invariant(self):
        import numpy as np
        from rhpr.gr.ae_warmup import water_filling, latent_spectrum
        self.assertAlmostEqual(water_filling([1, 1], 1)["water_level"], .5)
        row = water_filling([4, 1], .5)
        self.assertAlmostEqual(row["water_level"], 2.)
        self.assertEqual(row["active_modes"], 1)
        self.assertEqual(water_filling([1, 1], 0)["active_modes"], 0)
        self.assertEqual(water_filling([0, 0], 24)["active_modes"], 0)
        x = np.random.default_rng(7).normal(size=(200, 4)) * [1, .1, .01, .001]
        q, _ = np.linalg.qr(np.random.default_rng(9).normal(size=(4, 4)))
        a, b = latent_spectrum(x, 256, 3), latent_spectrum(x @ q + 9, 256, 3)
        np.testing.assert_allclose(a["eigenvalues"], b["eigenvalues"], atol=1e-12)
        self.assertEqual(a["aggregate_nominal_rate"]["rate_bits"], 24)
        self.assertEqual(a["single_book"]["rate_bits"], 8)
        self.assertEqual(a["effective_dimension_99pct"], b["effective_dimension_99pct"])
        for eigenvalues, rate in [([-1], 1), ([float('nan')], 1), ([1], -1)]:
            with self.assertRaises(ValueError):water_filling(eigenvalues, rate)

    def test_plateau_needs_minimum_and_consecutive_stability(self):
        from rhpr.gr.ae_warmup import SpectralPlateau
        monitor = SpectralPlateau(dict(kind='water_filling_plateau', min_steps=2, patience_probes=3), 10)
        def observe(step, count):
            return monitor.observe(step, {'aggregate_nominal_rate': {'active_modes': count}})
        self.assertFalse(observe(0, 3))
        self.assertFalse(observe(2, 3))
        self.assertFalse(observe(4, 2))
        self.assertFalse(observe(6, 2))
        self.assertTrue(observe(8, 2))
        with self.assertRaises(ValueError):SpectralPlateau(dict(kind='water_filling_plateau', min_steps=11, patience_probes=3), 10)


class StagingConfigTests(unittest.TestCase):
    def test_three_matched_branches_per_seed_and_strict_gpu_assignment(self):
        from rhpr.gr.ae_suite import assignments
        c = json.loads((ROOT / 'experiments/amazon_beauty_field_ae_warmup_rq_2t4.json').read_text())
        jobs = assignments(c, 2)
        self.assertEqual([[t['seed'] for t in j['trials']] for j in jobs], [[42]*3, [43]*3])
        self.assertEqual(c['quantizer']['latent_dim'], 64)
        self.assertEqual(c['warmup']['max_steps'], 5000)
        with self.assertRaises(RuntimeError):assignments(c, 1)
        c['trials'][0]['seed'] = 43
        with self.assertRaisesRegex(ValueError, 'one explicit pipeline seed'):assignments(c, 2)


@unittest.skipUnless(READY, "Requires pinned GRID and native dependencies")
class NativeStagingTests(unittest.TestCase):
    def test_parallel_pipeline_reuses_exact_ae_and_frozen_bn_buffers(self):
        script = r'''
import sys,json,numpy as np
from pathlib import Path
from rhpr.gr.grid import prepare_grid_source
from rhpr.gr.ae_suite import assignments,launch_workers
from rhpr.gr.study_parallel import atomic_json
root=Path(sys.argv[2]);source=prepare_grid_source(sys.argv[1],root/'grid')
data=root/'resources/data';data.mkdir(parents=True)
texts=[f"Title: T{i}; Brand: B{i%4}; Categories: ['Beauty','C{i%3}']; Price: {i+1}; " for i in range(32)]
(data/'catalog.json').write_text(json.dumps([dict(item_id=i,text=t) for i,t in enumerate(texts)]))
(data/'sequences.json').write_text(json.dumps([dict(user_id=0,train=[0,1,2],validation=3,test=4)]))
np.savez(root/'titles.npz',title=np.random.default_rng(7).normal(size=(32,8)).astype('float32'),title_observed=np.ones(32,dtype=bool))
policy=dict(mode='fixed_normalized',normalizer_floor=1e-6)
config=dict(seed=42,training=dict(device='cpu'),execution=dict(worker_devices=['cpu','cpu'],cpu_threads_per_worker=1),data={},
 representation=dict(kind='field_encoder',fields=['brand','categories','price','title'],field_dim=4),
 quantizer=dict(kind='grid_field_rqvae',latent_dim=8,num_hierarchies=3,codebook_size=4,batch_size=8,max_steps=4,init_buffer_size=8,initialization_steps=2,distance_backend='torch_cdist',diagnostics_interval_steps=2),
 warmup=dict(max_steps=6,loss_policy=policy,warmup_stopping=dict(kind='water_filling_plateau',min_steps=2,patience_probes=2)),
 trials=[dict(id=f'{mode}_s{seed}',seed=seed,freeze_policy=mode,loss_policy=policy) for mode in ['joint','ae','embeddings'] for seed in [42,43]])
output=root/'output';(output/'workers').mkdir(parents=True)
plan=dict(config=config,assignments=assignments(config,0),output=str(output),resources=str(root/'resources'),source=str(source),recipe=str(Path(sys.argv[1])/'configs/experiment/rqvae_train_flat.yaml'),parent_fields=str(root/'titles.npz'))
path=output/'workers/plan.json';atomic_json(path,plan)
summary=launch_workers(path)
assert len(summary['results'])==6 and not summary['failures']
assert set(summary['warmups'])=={'42','43'}
assert len({r['initial_state_sha256'] for r in summary['results'].values()})==2
for seed in [42,43]:
 rows=[summary['results'][f'{mode}_s{seed}'] for mode in ['joint','ae','embeddings']]
 warm=summary['warmups'][str(seed)]
 assert 2 <= warm['steps'] <= 6
 assert warm['stopping']['reason'] in ['spectral_plateau','max_steps']
 spectra=json.loads((output/f'warmup/seed{seed}/warmup-spectra.json').read_text())
 assert spectra[-1]['step']==warm['steps'] and spectra[-1]['aggregate_nominal_rate']['rate_bits']==6
 assert len({r['warmup']['checkpoint_sha256'] for r in rows})==1
 assert all(r['warmup']['checkpoint_sha256']==warm['checkpoint_sha256'] for r in rows)
 assert len({r['initial_full_state_sha256'] for r in rows})==1
 assert all(r['warmup']['neural_state_sha256']==r['initial_state_sha256'] for r in rows)
 books=[]
 for mode,row in zip(['joint','ae','embeddings'],rows):
  stage=row['warmup'];before,after=stage['module_hashes_before'],stage['module_hashes_after']
  assert stage['frozen_state_verified'] and row['steps']==row['reconstruction_updates']==4
  assert row['initialization_only_batches']>0
  directory=output/f'trials/{mode}_s{seed}'
  curves=json.loads((directory/'codebook-learning-curves.json').read_text())
  assert all('warmup_latent_drift_mse' in p for p in curves)
  for name in stage['frozen_modules']:assert before[name]==after[name]
  if mode=='ae':
   assert before==after
   assert curves[-1]['warmup_latent_drift_mse']==0
   assert curves[-1]['warmup_feature_drift_mse']==0
  else:
   assert before['encoder']!=after['encoder'] and before['decoder']!=after['decoder']
   assert curves[-1]['warmup_latent_drift_mse']>0
  if mode=='embeddings':assert curves[-1]['warmup_feature_drift_mse']==0
  if mode=='joint':assert before['feature_encoder']!=after['feature_encoder']
  with np.load(directory/'codebook-snapshots/step-00001.npz') as p:books.append(p['codebooks'])
 # First reconstruction update has matching centroids: initialization and its
 # first latent coordinates are identical despite different gradient masks.
 for b in books[1:]:np.testing.assert_array_equal(b,books[0])
'''
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([sys.executable, '-c', script, SOURCE, directory],
                capture_output=True, text=True, timeout=180)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':unittest.main()
