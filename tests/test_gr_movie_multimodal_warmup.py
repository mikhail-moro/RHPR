"""Warm-up transfer, untouched books, and the one-fit experiment contract."""
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'experiments/movielens100k_text_ae_warmup.json'


class MultimodalWarmupTest(unittest.TestCase):
    def test_completed_kernel_input_is_hash_checked_and_accepts_legacy_mount(self):
        from rhpr.gr.movie_multimodal import frozen_control_input
        with tempfile.TemporaryDirectory(dir='/tmp') as tmp:
            parent=Path(tmp);data=parent/'shared-modalities';data.mkdir()
            (parent/'run-manifest.json').write_text('{"git_sha":"parent"}')
            (data/'features.npz').write_bytes(b'exact frozen vectors')
            (data/'source-ids.json').write_text('[1,2]')
            spec={'path':str(parent/'absent'),'legacy_path':str(parent),'parent_git_sha':'parent'}
            for name,key in [(parent/'run-manifest.json','parent_manifest_sha256'),(data/'features.npz','features_sha256'),(data/'source-ids.json','source_ids_sha256')]:
                spec[key]=hashlib.sha256(name.read_bytes()).hexdigest()
            self.assertEqual(frozen_control_input(spec),data)
            (data/'features.npz').write_bytes(b'changed')
            with self.assertRaises(ValueError):frozen_control_input(spec)

    def test_one_matched_text_seed_and_frozen_inputs(self):
        from rhpr.gr.movie_suite import validate, movie_assignments
        config = json.loads(CONFIG.read_text())
        validate(config)
        jobs = movie_assignments(config, 2)
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0]['trials'], config['trials'])
        self.assertEqual(jobs[0]['device'], 'cuda:0')
        with self.assertRaises(RuntimeError):
            movie_assignments(config, 0)
        previous = json.loads((ROOT/'experiments/movielens100k_multimodal_2t4.json').read_text())
        for key in ['training', 'generator', 'data', 'movie_input', 'representation', 'fields']:
            self.assertEqual(config[key], previous[key])
        for mutate in ['extra_seed', 'extra_trial', 'other_modality', 'no_warmup', 'more_rq_steps', 'other_vectors']:
            bad = copy.deepcopy(config)
            if mutate == 'extra_seed': bad['replicate_seeds'].append(101)
            elif mutate == 'extra_trial': bad['trials'].append(copy.deepcopy(bad['trials'][0]))
            elif mutate == 'other_modality': bad['trials'][0]['modalities'] = ['text', 'poster']
            elif mutate == 'no_warmup': bad['quantizer'].pop('ae_warmup')
            elif mutate == 'more_rq_steps': bad['quantizer']['max_steps'] = 5000
            else: bad['modality_input']['features_sha256'] = '0'*64
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                validate(bad)

    @unittest.skipUnless(os.environ.get('RHPR_GRID_SOURCE'), 'requires pinned native GRID')
    def test_continuous_ae_then_native_rq_and_tiny_tiger(self):
        script = r'''
import json,sys,torch,numpy as np
from pathlib import Path
from unittest.mock import patch
from rhpr.gr.grid import prepare_grid_source,build_grid_generator
from rhpr.gr.rqvae import build_rqvae
from rhpr.gr.multimodal_rqvae import normalize_modalities
from rhpr.gr.multimodal_warmup import warmup_autoencoder,neural_state
from rhpr.gr.sid_study import state_digest
from rhpr.gr.movie_multimodal import run_multimodal_trial
from rhpr.gr.movie_data import TemporalUser
root=Path(sys.argv[1]);original=Path(sys.argv[2]);source=prepare_grid_source(original,root/'grid')
recipe=original/'configs/experiment/rqvae_train_flat.yaml';torch.set_num_threads(1)
c=json.loads(Path(sys.argv[3]).read_text())
# Exact prior quantizer initialization despite adding AE support.
torch.manual_seed(42)
full,_=build_rqvae(source,recipe,512,c['quantizer'],modality_schema=dict(text=768,poster=768,plot=768,people=128),active_modalities=['text'])
assert state_digest(dict(full.named_parameters()))==c['quantizer']['expected_initial_state_sha256'];del full
spec=dict(num_hierarchies=3,codebook_size=16,batch_size=16,max_steps=8,init_buffer_size=16,initialization_steps=1,distance_backend='torch_cdist',latent_dim=4,head_dim=4,head_hidden=8,diagnostics_interval_steps=4,ae_warmup=dict(max_steps=8,probe_interval=2,stopping=dict(kind='water_filling_plateau',min_steps=2,patience_probes=2)))
schema=dict(text=8,poster=6,plot=5,people=4)
arrays={k:np.random.default_rng(d).normal(size=(16,d)).astype('float32') for k,d in schema.items()}
masks={k:np.ones(16,dtype=bool) for k in schema}
norm,stats=normalize_modalities(arrays,masks)
torch.manual_seed(42);model,_=build_rqvae(source,recipe,16,spec,modality_schema=schema,active_modalities=['text'])
tensors={k:torch.from_numpy(v) for k,v in norm.items()};tensors.update({k+'_observed':torch.from_numpy(v) for k,v in masks.items()});tensors['_item_indices']=torch.arange(16)
# Continuous AE must never invoke the quantizer, even once.
with patch.object(model,'forward',side_effect=AssertionError('quantization used during AE')):
    report=warmup_autoencoder(model,tensors,spec,42,'cpu',root/'only-ae',stats)
assert report['codebooks_unchanged'] and report['neural_state_sha256']!=report['initial_neural_state_sha256']
assert report['neural_state_sha256']==state_digest(neural_state(model))
assert not any(book.is_initialized for book in model.quantization_layer_list)
assert 4<=report['steps']<=8
# Full path: same model survives AE/native initialization/RQ and then 20 tiny epochs.
c['quantizer']=spec;c['generator'].update(d_model=8,d_ff=16,d_kv=4,num_heads=2,encoder_layers=1,decoder_layers=1,dropout=0.,num_user_bins=0,codebook_size=16,beam_width=2)
c['training'].update(device='cpu',precision='fp32',evaluation_precision='fp32',batch_size=8,physical_batch_size=8,evaluation_batch_size=2,max_seconds=120)
data=root/'modalities';data.mkdir();np.savez_compressed(data/'features.npz',**arrays,**{k+'_observed':v for k,v in masks.items()})
(data/'source-ids.json').write_text(json.dumps(list(range(16))));(data/'provenance.json').write_text('{}')
users=[TemporalUser(1,(0,1,2,3),(1,1,2,2),4,3,5,4),TemporalUser(2,(6,7,8,9),(1,1,2,2),10,3,11,4)]
dummy=np.zeros((16,4),dtype=np.int64);dummy[:,0]=np.arange(16)
g=build_grid_generator(dummy,c['generator'],42,'cpu',source);initial=state_digest(dict(g.named_parameters()));del g
plan=dict(output=str(root/'suite'),modality_root=str(data),source_ids=list(range(16)),original=str(original),initial_state_sha256_by_seed={'42':initial})
r=run_multimodal_trial(dict(id='seed42_text_ae',seed=42,stage=1,modalities=['text']),c,plan,users,source,'cpu')
assert r['selection']['training_epochs']==20 and r['selection']['validation_checks']==4
q=json.loads((root/'suite/trials/seed42_text_ae/quantizer.json').read_text())
assert q['steps']==8 and q['reconstruction_batches']>0
assert q['warmup']['fresh_rq_optimizer_verified'] and q['warmup']['neural_transfer_verified']
assert q['initialization_latent_audit']['max_absolute_difference']<1e-5
# Native initialization uses a separate SGD and does not advance Adagrad's step.
assert q['reconstruction_batches']==8 and q['batches_without_reconstruction']==1
assert (root/'suite/trials/seed42_text_ae/test-predictions.npz').exists()
'''
        with tempfile.TemporaryDirectory(dir='/tmp') as tmp:
            p = subprocess.run([sys.executable, '-c', script, tmp, os.environ['RHPR_GRID_SOURCE'], str(CONFIG)], capture_output=True, text=True, timeout=180)
            self.assertEqual(p.returncode, 0, p.stdout+p.stderr)


if __name__ == '__main__':
    unittest.main()
