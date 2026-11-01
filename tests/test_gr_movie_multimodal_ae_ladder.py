"""Full cumulative ladder uses the proven AE recipe and seed-matched controls."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
CONFIG=ROOT/'experiments/movielens100k_multimodal_ae_2t4.json'


class MovieMultimodalAELadderTest(unittest.TestCase):
    def test_twelve_matched_staged_trials_on_two_workers(self):
        from rhpr.gr.movie_suite import validate,movie_assignments
        c=json.loads(CONFIG.read_text());validate(c)
        old=json.loads((ROOT/'experiments/movielens100k_multimodal_2t4.json').read_text())
        single=json.loads((ROOT/'experiments/movielens100k_text_ae_warmup.json').read_text())
        for key in ['generator','training','data','movie_input','representation','fields','trials','execution','replicate_seeds','expected_initial_state_sha256_by_seed','plot_encoder']:
            self.assertEqual(c[key],old[key],key)
        for key,value in old['quantizer'].items():self.assertEqual(c['quantizer'][key],value)
        self.assertEqual(c['quantizer']['ae_warmup'],single['quantizer']['ae_warmup'])
        self.assertEqual(c['modality_input'],single['modality_input'])
        self.assertEqual(c['kaggle'],single['kaggle'])
        jobs=movie_assignments(c,2)
        self.assertEqual([len(x['trials']) for x in jobs],[6,6])
        self.assertEqual(len({t['id'] for job in jobs for t in job['trials']}),12)
        with self.assertRaises(RuntimeError):movie_assignments(c,1)
        for kind in ['missing_seed','single_seed_hash','bad_warmup','missing_trial','wrong_channel','changed_vectors']:
            bad=copy.deepcopy(c)
            if kind=='missing_seed':bad['quantizer']['expected_initial_state_sha256_by_seed'].pop('101')
            elif kind=='single_seed_hash':bad['quantizer']['expected_initial_state_sha256']='0'*64
            elif kind=='bad_warmup':bad['quantizer']['ae_warmup']['max_steps']=3000
            elif kind=='missing_trial':bad['trials'].pop()
            elif kind=='wrong_channel':bad['trials'][-1]['modalities']=['text']
            else:bad['modality_input']['features_sha256']='0'*64
            with self.subTest(kind=kind),self.assertRaises(ValueError):validate(bad)

    @unittest.skipUnless(os.environ.get('RHPR_GRID_SOURCE'),'requires pinned native GRID')
    def test_all_seeds_and_active_heads_warmup_transfer_and_native_codes(self):
        script=r'''
import json,sys,torch,numpy as np
from pathlib import Path
from rhpr.gr.grid import prepare_grid_source
from rhpr.gr.rqvae import build_rqvae
from rhpr.gr.sid_study import state_digest
from rhpr.gr.multimodal_rqvae import fit_multimodal_rqvae
root=Path(sys.argv[1]);original=Path(sys.argv[2]);source=prepare_grid_source(original,root/'grid');recipe=original/'configs/experiment/rqvae_train_flat.yaml'
c=json.loads(Path(sys.argv[3]).read_text());torch.set_num_threads(1)
schema=dict(text=768,poster=768,plot=768,people=128)
for seed in c['replicate_seeds']:
 torch.manual_seed(seed);m,_=build_rqvae(source,recipe,512,c['quantizer'],modality_schema=schema,active_modalities=list(schema))
 assert state_digest(dict(m.named_parameters()))==c['quantizer']['expected_initial_state_sha256_by_seed'][str(seed)];del m
schema=dict(text=8,poster=6,plot=5,people=4)
values={k:np.random.default_rng(d).normal(size=(16,d)).astype('float32') for k,d in schema.items()};masks={k:np.ones(16,dtype=bool) for k in schema};masks['plot'][:3]=False
spec=dict(num_hierarchies=3,codebook_size=16,batch_size=16,max_steps=8,init_buffer_size=16,initialization_steps=1,distance_backend='torch_cdist',latent_dim=4,head_dim=4,head_hidden=8,diagnostics_interval_steps=4,ae_warmup=dict(max_steps=8,probe_interval=2,stopping=dict(kind='water_filling_plateau',min_steps=2,patience_probes=2)))
pins={}
for seed in c['replicate_seeds']:
 torch.manual_seed(seed);m,_=build_rqvae(source,recipe,16,spec,modality_schema=schema,active_modalities=list(schema));pins[str(seed)]=state_digest(dict(m.named_parameters()));del m
spec['expected_initial_state_sha256_by_seed']=pins
for seed in c['replicate_seeds']:
 for stage in range(1,5):
  active=list(schema)[:stage];p=root/f'seed{seed}_stage{stage}'
  codes,r=fit_multimodal_rqvae(values,masks,spec,seed,'cpu',source,recipe,p,active)
  assert r['quantizer_initial_state_sha256']==pins[str(seed)]
  assert codes.shape==(16,3) and r['steps']==8 and r['reconstruction_batches']==8 and r['batches_without_reconstruction']==1
  assert r['warmup']['codebooks_unchanged'] and r['warmup']['neural_transfer_verified'] and r['warmup']['fresh_rq_optimizer_verified']
  assert r['initialization_latent_audit']['max_absolute_difference']<1e-5
  assert set(r['head_reconstruction_mse'])==set(active) and all(np.isfinite(x) for x in r['head_reconstruction_mse'].values())
try:fit_multimodal_rqvae(values,masks,spec,999,'cpu',source,recipe,root/'wrong-seed',['text'])
except ValueError as e:assert 'not pinned' in str(e)
else:raise AssertionError('unregistered seed accepted')
'''
        with tempfile.TemporaryDirectory(dir='/tmp') as tmp:
            p=subprocess.run([sys.executable,'-c',script,tmp,os.environ['RHPR_GRID_SOURCE'],str(CONFIG)],capture_output=True,text=True,timeout=180)
            self.assertEqual(p.returncode,0,p.stdout+p.stderr)


if __name__=='__main__':unittest.main()
