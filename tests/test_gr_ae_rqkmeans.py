"""Frozen-latent residual KMeans and the complete two-worker TIGER path."""
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
CLUSTER_READY = all(importlib.util.find_spec(n) for n in ("numpy", "sklearn", "threadpoolctl"))
NATIVE_READY = SOURCE and CLUSTER_READY and all(importlib.util.find_spec(n) for n in ("torch", "hydra", "lightning", "transformers"))


class ConfigTests(unittest.TestCase):
    def test_only_two_kmeans_indices_change_against_completed_generator_control(self):
        from rhpr.gr.generator_suite import assignments
        c=json.loads((ROOT/'experiments/amazon_beauty_field_ae_rqkmeans_tiger_2t4.json').read_text())
        old=json.loads((ROOT/'experiments/amazon_beauty_field_warmup_tiger_2t4.json').read_text())
        for key in ['seed','data','generator','training','evaluation','execution','parent']:
            self.assertEqual(c[key],old[key])
        self.assertEqual(c['quantizer']['kind'],'ae_residual_kmeans')
        self.assertFalse(c['quantizer']['normalize_residuals'])
        self.assertFalse(c['quantizer']['normalize_latents'])
        self.assertEqual([[t['id'] for t in a['trials']] for a in assignments(c,2)],
            [['ae_rqkm_s42'],['ae_rqkm_s43']])
        self.assertEqual(set(c['controls']['runs']),{'ae_s42','ae_s43'})
        self.assertTrue(c['expected_initial_state_sha256'])
        self.assertTrue(c['expected_train_cases_sha256'])


@unittest.skipUnless(CLUSTER_READY,'Requires NumPy and sklearn')
class ResidualKMeansTests(unittest.TestCase):
    def test_fixed_scale_greedy_codes_replay_without_mutating_latents_or_global_rng(self):
        import numpy as np
        from rhpr.gr.ae_rqkmeans import fit_residual_kmeans
        z=np.random.default_rng(5).normal(size=(80,6)).astype('float32')*9+25
        before=z.copy();rng=np.random.get_state()
        spec=dict(num_hierarchies=3,codebook_size=4,max_iter=100,n_init=1,cpu_threads=1,tol=1e-4,normalize_latents=False,normalize_residuals=False)
        codes,books,report=fit_residual_kmeans(z,spec,42)
        np.testing.assert_array_equal(z,before)
        after=np.random.get_state();self.assertEqual(rng[0],after[0]);np.testing.assert_array_equal(rng[1],after[1]);self.assertEqual(rng[2:],after[2:])
        residual=z.copy();mse=float(np.square(residual).mean())
        for level in range(3):
            distance=np.square(residual[:,None,:]-books[level][None,:,:]).sum(2)
            np.testing.assert_array_equal(distance.argmin(1),codes[:,level])
            residual-=books[level][codes[:,level]]
            self.assertLess(float(np.square(residual).mean()),mse);mse=float(np.square(residual).mean())
        c2,b2,_=fit_residual_kmeans(z,spec,42)
        np.testing.assert_array_equal(codes,c2);np.testing.assert_array_equal(books,b2)
        # Raw-scale preservation:the first book still includes the large offset.
        self.assertGreater(float(np.mean(books[0])),10)
        self.assertEqual(report['scope'],'full catalogue raw AE latents; sequential residual Lloyd; no neural updates')
        with self.assertRaisesRegex(ValueError,'raw latent'):fit_residual_kmeans(z,{**spec,'normalize_residuals':True},42)
        with self.assertRaisesRegex(ValueError,'finite'):fit_residual_kmeans(z*np.nan,spec,42)
        with self.assertRaisesRegex(ValueError,'positive integer'):fit_residual_kmeans(z,{**spec,'max_iter':0},42)


@unittest.skipUnless(NATIVE_READY,'Requires pinned GRID and native dependencies')
class NativePipelineTests(unittest.TestCase):
    def test_two_frozen_aes_to_cpu_books_then_two_real_tiger_workers(self):
        script=r'''
import sys,json,numpy as np
from pathlib import Path
from unittest.mock import patch
from rhpr.gr.generator_suite import run_generator_suite,prepare_indices
from rhpr.gr.data import file_sha256
from rhpr.metrics import validate_metrics
root=Path(sys.argv[2]);data=root/'resources/data';data.mkdir(parents=True)
(data/'catalog.json').write_text(json.dumps([dict(item_id=i,text=f'item{i}') for i in range(24)]))
(data/'sequences.json').write_text(json.dumps([dict(user_id=i,train=[i,i+1,i+2,i+3],validation=i+4,test=i+5) for i in range(3)]))
data_spec=dict(history_items=2,sha256={n:file_sha256(data/n) for n in ['catalog.json','sequences.json']})
parent=root/'parent';parent.mkdir()
(parent/'run-manifest.json').write_text(json.dumps(dict(git_sha='parent',config=dict(data=data_spec))))
(parent/'input-provenance.json').write_text(json.dumps(dict(original_item_ids=list(range(24)))))
def pin(path):return dict(path=path.relative_to(parent).as_posix(),sha256=file_sha256(path),size_bytes=path.stat().st_size)
trials=[]
for seed in [42,43]:
 d=parent/f'warmup/seed{seed}';d.mkdir(parents=True)
 np.save(d/'latent-embeddings.npy',np.random.default_rng(seed).normal(size=(24,6)).astype('float32'))
 (d/'autoencoder.json').write_text(json.dumps(dict(seed=seed,quantization_enabled=False,steps=3500,latent_dim=6,checkpoint_sha256=f'ae{seed}')))
 trials.append(dict(id=f'ae_rqkm_s{seed}',quantizer_seed=seed,freeze_policy='ae',ae_latents=pin(d/'latent-embeddings.npy'),ae_report=pin(d/'autoencoder.json'),ae_steps=3500,ae_checkpoint_sha256=f'ae{seed}'))
config=dict(seed=42,name='fixture',kind='gr_frozen_sid_generator_suite',data=data_spec,
 resources=dict(path=str(root/'resources'),manifest_sha256='fixture',scratch_path=str(root/'scratch')),
 parent=dict(kernel='owner/fixture',path=str(parent),source_code_sha='parent',files={n:{k:v for k,v in pin(parent/n).items() if k!='path'} for n in ['run-manifest.json','input-provenance.json']}),
 quantizer=dict(kind='ae_residual_kmeans',num_hierarchies=3,codebook_size=8,latent_dim=6,n_init=1,max_iter=100,tol=1e-4,cpu_threads=1,normalize_latents=False,normalize_residuals=False),
 generator=dict(d_model=8,d_ff=16,d_kv=4,num_heads=2,encoder_layers=1,decoder_layers=1,dropout=.1,mlp_layers=None,num_user_bins=8,separator_token=False,codebook_size=8,beam_width=2,should_check_prefix=False),
 training=dict(device='cpu',sampling='all_train_prefixes',batch_size=4,physical_batch_size=2,learning_rate=.001,weight_decay=.0001,grad_clip_norm=1.,max_epochs=2,min_epochs=2,patience=2,patience_epochs=2,validation_interval=2,validate_first_epoch=False,evaluation_batch_size=2,max_seconds=120,precision='fp32',evaluation_precision='fp32'),
 execution=dict(worker_devices=['cpu','cpu'],cpu_threads_per_worker=1),trials=trials,controls={'reused':True})
output=root/'out';output.mkdir()
with patch('rhpr.offline_runtime.resolve_dataset_directory',return_value=root/'resources'),patch('rhpr.gr.resources.verify_resources',return_value={}),patch('rhpr.gr.resources.unpack_grid',return_value=Path(sys.argv[1])):
 metrics=validate_metrics(run_generator_suite(config,output))
assert metrics['runs']==2 and metrics['quantizer_retrained'] and not metrics['autoencoder_retrained']
summary=json.loads((output/'suite-results.json').read_text());assert len(summary['results'])==2 and not summary['failures']
assert len({r['initial_state_sha256'] for r in summary['results'].values()})==1
for t in trials:
 d=output/'indices'/t['id'];p=json.loads((d/'index-provenance.json').read_text())
 assert p['quantizer_retrained'] and not p['autoencoder_retrained']
 report=json.loads((d/'kmeans.json').read_text());assert len(report['levels'])==3 and report['ae_checkpoint_sha256']==t['ae_checkpoint_sha256']
 assert np.load(d/'codebooks.npy').shape==(3,8,6)
 assert np.load(d/'semantic-ids.npy').shape==(24,4)
 run=summary['results'][t['id']];assert run['selection']['epochs']==2 and run['test_evaluations']==1
 assert run['selection']['model_examples_consumed']==18
 assert (output/'trials'/t['id']/'generator-best.pt').is_file()
assert (output/'controls.json').is_file()
assert not list(output.rglob('quantizer.pt')) and not list(output.rglob('autoencoder.pt'))
# Valid bytes but wrong registered AE identity fail before fitting any books.
bad={**config,'trials':[{**trials[0],'ae_steps':3000}]}
with patch('rhpr.gr.ae_rqkmeans.fit_residual_kmeans',side_effect=AssertionError('fit called')):
 try:prepare_indices(bad,parent,list(range(24)),root/'bad')
 except ValueError as e:assert 'AE latent source/report' in str(e)
 else:raise AssertionError('wrong AE accepted')
# Case/init guards compare with completed control before workers start.
with patch('rhpr.offline_runtime.resolve_dataset_directory',return_value=root/'resources'),patch('rhpr.gr.resources.verify_resources',return_value={}),patch('rhpr.gr.resources.unpack_grid',return_value=Path(sys.argv[1])),patch('rhpr.gr.ae_suite.launch_workers',side_effect=AssertionError('workers started')):
 for key in ['expected_train_cases_sha256','expected_initial_state_sha256']:
  out=root/key;out.mkdir()
  try:run_generator_suite({**config,key:'bad'},out)
  except ValueError as e:assert 'completed TIGER control' in str(e)
  else:raise AssertionError('wrong completed control accepted')
'''
        with tempfile.TemporaryDirectory() as directory:
            result=subprocess.run([sys.executable,'-c',script,SOURCE,directory],capture_output=True,text=True,timeout=180)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)


if __name__=='__main__':unittest.main()
