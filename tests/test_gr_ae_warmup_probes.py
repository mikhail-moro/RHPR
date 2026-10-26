"""TRAIN-only proxies, observation-invariant snapshots and two real AE workers."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
SOURCE=os.environ.get('RHPR_GRID_SOURCE')
READY=SOURCE and all(importlib.util.find_spec(n) for n in ['numpy','torch','hydra','lightning','transformers','sklearn','scipy','matplotlib'])


class ContractTests(unittest.TestCase):
    def test_fixed_stop_diagnostics_have_no_tiger_or_adaptive_stopping(self):
        from rhpr.gr.ae_warmup_probe import validate_probe
        c=json.loads((ROOT/'experiments/amazon_beauty_field_ae_warmup_probes_2t4.json').read_text())
        validate_probe(c)
        self.assertEqual(c['quantizer']['snapshot_steps'],[0,100,250,500,1000,2000,3500,5000])
        for key in ['generator','evaluation']:self.assertNotIn(key,c)
        for bad in [[0,1,1,5000],[5000,0],[-1,5000],[True,5000]]:
            with self.assertRaises(ValueError):validate_probe({**c,'quantizer':{**c['quantizer'],'snapshot_steps':bad}})
        with self.assertRaises(ValueError):validate_probe({**c,'generator':{}})


@unittest.skipUnless(importlib.util.find_spec('numpy') and importlib.util.find_spec('scipy'),'Requires NumPy/SciPy')
class ProxyTests(unittest.TestCase):
    def test_train_only_permutation_invariant_with_known_groups_and_missing_values(self):
        import numpy as np
        from rhpr.gr.index_proxies import IndexProxies
        class User:
            def __init__(self,train):self.train=train
            @property
            def validation(self):raise AssertionError('validation consulted')
            @property
            def test(self):raise AssertionError('test consulted')
        users=[User([0,1]) for _ in range(3)]+[User([2,3]) for _ in range(3)]
        proxy=IndexProxies(users,[1,1,4,4],[True]*4)
        a=proxy.first(np.array([0,0,1,1]));b=proxy.first(np.array([9,9,3,3]))
        self.assertEqual(a,b);self.assertAlmostEqual(a['train_prefix_coherence'],1.)
        self.assertEqual(a['price_r2_trim99'],1.);self.assertGreater(a['train_proxy_gain_bits'],0)
        self.assertIsNone(proxy.first(np.arange(4))['train_prefix_coherence'])
        self.assertIsNone(IndexProxies(users,[0,0,0,0],[False]*4).first(np.zeros(4,dtype=int))['price_r2'])
        with self.assertRaises(ValueError):proxy.first(np.array([1,2]))
        with self.assertRaises(ValueError):IndexProxies([User([99])],[1],[True])


@unittest.skipUnless(READY,'Requires pinned GRID and native dependencies')
class NativeTests(unittest.TestCase):
    def test_snapshot_observation_does_not_change_training_and_full_probe_has_no_generator(self):
        script=r'''
import sys,json,numpy as np,torch
from pathlib import Path
from unittest.mock import patch
from rhpr.gr.grid import prepare_grid_source
from rhpr.gr.field_autoencoder import fit_field_autoencoder
from rhpr.gr.ae_warmup_probe import run_warmup_probe
from rhpr.gr.data import file_sha256
from rhpr.metrics import validate_metrics
root=Path(sys.argv[2]);source=prepare_grid_source(sys.argv[1],root/'grid');torch.set_num_threads(1)
texts=[f"Title: T{i}; Brand: B{i%4}; Categories: ['Beauty','C{i%3}']; Price: {i+1}; " for i in range(24)]
titles=np.random.default_rng(7).normal(size=(24,8)).astype('float32')
spec=dict(kind='continuous_field_autoencoder',num_hierarchies=3,codebook_size=4,latent_dim=6,batch_size=8,max_steps=4,init_buffer_size=8,initialization_steps=2,distance_backend='torch_cdist',diagnostics_interval_steps=2,loss_policy={'mode':'fixed_normalized'})
rep=dict(kind='field_encoder',fields=['brand','categories','price','title'],field_dim=4,title={'dimension':8})
recipe=Path(sys.argv[1])/'configs/experiment/rqvae_train_flat.yaml'
for name,extra in [('plain',{}),('probes',{'snapshot_steps':[0,1,2,4]})]:
 fit_field_autoencoder(texts,rep,{**spec,**extra},42,'cpu',source,recipe,root/name,title_embeddings=titles)
a=torch.load(root/'plain/autoencoder.pt',weights_only=False);b=torch.load(root/'probes/autoencoder.pt',weights_only=False)
assert all(torch.equal(v,b['model'][k]) for k,v in a['model'].items())
assert torch.equal(a['rng_cpu'],b['rng_cpu']) and torch.equal(a['data_rng'],b['data_rng'])
for k,entry in a['optimizer']['state'].items():
 for n,v in entry.items():
  if torch.is_tensor(v):assert torch.equal(v,b['optimizer']['state'][k][n])
  else:assert v==b['optimizer']['state'][k][n]
np.testing.assert_array_equal(np.load(root/'plain/latent-embeddings.npy'),np.load(root/'probes/latent-embeddings.npy'))
np.testing.assert_array_equal(np.load(root/'probes/checkpoints/step-00004/latent-embeddings.npy'),np.load(root/'plain/latent-embeddings.npy'))
data=root/'resources/data';data.mkdir(parents=True)
(data/'catalog.json').write_text(json.dumps([dict(item_id=i,text=texts[i]) for i in range(24)]))
(data/'sequences.json').write_text(json.dumps([dict(user_id=i,train=[i,i+1,i+2,i+3],validation=i+4,test=i+5) for i in range(6)]))
parent=root/'parent';parent.mkdir()
np.savez_compressed(parent/'catalogue-fields.npz',title=titles,title_observed=np.ones(24,dtype=bool))
(parent/'run-manifest.json').write_text(json.dumps({'git_sha':'parent'}))
config=dict(kind='gr_ae_warmup_probe',name='fixture',seed=42,compare_initialization_by_seed=True,
 data=dict(history_items=2,sha256={n:file_sha256(data/n) for n in ['catalog.json','sequences.json']}),
 resources=dict(path=str(root/'resources'),manifest_sha256='fixture',scratch_path=str(root/'scratch')),
 parent=dict(kernel='owner/fixture',path=str(parent),source_code_sha='parent',files={n:dict(sha256=file_sha256(parent/n),size_bytes=(parent/n).stat().st_size) for n in ['run-manifest.json','catalogue-fields.npz']}),
 representation=rep,quantizer={**spec,'snapshot_steps':[0,1,2,4]},
 kmeans=dict(kind='ae_residual_kmeans',num_hierarchies=3,codebook_size=4,latent_dim=6,n_init=1,max_iter=50,tol=1e-4,cpu_threads=1,normalize_latents=False,normalize_residuals=False),
 training=dict(device='cpu',precision='fp32'),execution=dict(worker_devices=['cpu','cpu'],cpu_threads_per_worker=1),
 trials=[dict(id=f'ae_s{s}',seed=s,loss_policy=spec['loss_policy']) for s in [42,43]])
output=root/'out';output.mkdir()
with patch('rhpr.offline_runtime.resolve_dataset_directory',return_value=root/'resources'),patch('rhpr.gr.resources.verify_resources',return_value={}),patch('rhpr.gr.resources.unpack_grid',return_value=Path(sys.argv[1])),patch('rhpr.gr.grid.build_grid_generator',side_effect=AssertionError('generator called')),patch('rhpr.gr.experiment._evaluate',side_effect=AssertionError('retrieval called')):
 metrics=validate_metrics(run_warmup_probe(config,output))
assert metrics['index_probes']==8 and metrics['generator_trained'] is False and metrics['test_evaluations']==0
assert (output/'warmup-proxies.png').is_file() and (output/'probe-curves.csv').is_file()
rows=json.loads((output/'probe-curves.json').read_text());assert len(rows)==8
for r in rows:
 d=output/'trials'/f'ae_s{r["seed"]}'/'checkpoints'/f'step-{r["step"]:05d}'
 assert r['codes_sha256']==file_sha256(d/'semantic-codes.npy')
 assert np.load(d/'codebooks.npy').shape==(3,4,6)
assert not list(output.rglob('generator*.pt')) and not list(output.rglob('test*.json'))
'''
        with tempfile.TemporaryDirectory() as directory:
            r=subprocess.run([sys.executable,'-c',script,SOURCE,directory],capture_output=True,text=True,timeout=180)
            self.assertEqual(r.returncode,0,r.stdout+r.stderr)


if __name__=='__main__':unittest.main()
