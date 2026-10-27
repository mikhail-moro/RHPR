"""Identical mapping control, independent neural RNG, and rank-preserving oracle."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from rhpr.gr.generator_suite import assignments, generator_config, training_manifest
from rhpr.gr.sid_seed_oracle import pair_scores, saved_same_index_oracle

ROOT = Path(__file__).resolve().parents[1]
SOURCE = os.environ.get('RHPR_GRID_SOURCE')


class SeedOracleTests(unittest.TestCase):
    def config(self):
        return json.loads((ROOT/'experiments/amazon_beauty_sid_seed_oracle_2t4.json').read_text())

    def test_fixed_data_order_distinct_neural_seeds_and_existing_protocol(self):
        c = self.config()
        old = json.loads((ROOT/'experiments/amazon_beauty_sid_context_screen_2t4.json').read_text())
        for key in ['seed','data','generator','training','frozen_indices','expected_train_cases_sha256']:
            self.assertEqual(c[key], old[key])
        self.assertEqual([j['trials'][0]['generator_seed'] for j in assignments(c,2)], [42,43])
        self.assertEqual(c['trials'][0]['expected_initial_state_sha256'], c['expected_initial_state_sha256'])
        self.assertNotEqual(*[t['expected_initial_state_sha256'] for t in c['trials']])
        for t in c['trials']:
            self.assertTrue(t['use_original_paths'])
            self.assertEqual(t['quantizer_seed'],42)
            self.assertEqual(generator_config(c,t)['seed'],t['generator_seed'])
        self.assertEqual(c['seed'],42)
        class User:
            user_id=0
            train=(0,1,2,3)
        before=training_manifest([User()],c)
        altered=copy.deepcopy(c);altered['trials'][1]['generator_seed']=999
        self.assertEqual(before,training_manifest([User()],altered))
        for key,value in [('generator_seed',True),('generator_seed',-1)]:
            bad=copy.deepcopy(c);bad['trials'][1][key]=value
            with self.assertRaises(ValueError):assignments(bad,2)
        bad=copy.deepcopy(c);bad['trials'][1]['use_original_paths']=False
        with self.assertRaises(ValueError):assignments(bad,2)

    def test_oracle_preserves_rank_and_counts_overlap(self):
        a=np.array([1,8,999,999,10,2]);b=np.array([2,3,4,999,11,2])
        r=pair_scores(a,b)
        expected=np.array([1,1/np.log2(4),1/np.log2(5),0,1/np.log2(11),1/np.log2(3)])
        self.assertAlmostEqual(r['oracle_ndcg10'],expected.mean())
        self.assertEqual([r[k] for k in ['a_only_hits','b_only_hits','both_hits','both_miss']],[1,1,3,1])
        same=pair_scores(a,a)
        self.assertEqual(same['absolute_gain'],0)
        self.assertFalse(pair_scores([999],[999])['relative_gain_defined'])

    @unittest.skipUnless(SOURCE,'Set RHPR_GRID_SOURCE for real native seed fits')
    def test_native_two_seeds_same_sid_train_and_saved_oracle(self):
        script=r'''
import json,sys
from pathlib import Path
import numpy as np
import torch
from rhpr.gr.sid import SidIndex
from rhpr.gr.grid import build_grid_generator
from rhpr.gr.generator_suite import run_trial,generator_config
from rhpr.gr.sid_study import state_digest
from rhpr.gr.sid_seed_oracle import saved_same_index_oracle
torch.set_num_threads(1)
root=Path(sys.argv[2]);source=sys.argv[1]
index=SidIndex(np.array([[i//6,0,0] for i in range(24)]),width=8)
gen=dict(d_model=8,d_ff=16,d_kv=4,num_heads=2,encoder_layers=1,decoder_layers=1,
 dropout=.1,mlp_layers=None,num_user_bins=8,separator_token=False,codebook_size=8,beam_width=2,should_check_prefix=False)
cfg=dict(seed=42,data=dict(history_items=2),quantizer=dict(kind='frozen_full_sids',codebook_size=8),
 generator=gen,training=dict(device='cpu',sampling='all_train_prefixes',batch_size=4,physical_batch_size=2,
 learning_rate=.001,weight_decay=.0001,grad_clip_norm=1.,max_epochs=2,min_epochs=2,
 patience=2,patience_epochs=2,validation_interval=1,validate_first_epoch=False,
 evaluation_batch_size=2,max_seconds=120,precision='fp32',evaluation_precision='fp32'),
 evaluation=dict(test_after_selection=False,save_validation_predictions=True))
class User:
 def __init__(self,i):self.user_id=i;self.train=(i,i+1,i+2,i+3);self.validation=i+4
 @property
 def test(self):raise AssertionError('test data accessed')
users=[User(i) for i in range(3)]
cfg['trials']=[dict(id=f'seed{s}',generator_seed=s,quantizer_seed=42,freeze_policy='frozen_full_paths',use_original_paths=True) for s in [42,43]]
initials=[]
for trial in cfg['trials']:
 name=trial['id'];d=root/'indices'/name;d.mkdir(parents=True)
 np.save(d/'semantic-ids.npy',index.sids)
 (d/'index-provenance.json').write_text(json.dumps(dict(preserve_full_sids=True)))
 m=build_grid_generator(index.sids,gen,trial['generator_seed'],'cpu',source)
 initial=state_digest(dict(m.named_parameters()));initials.append(initial)
 result=run_trial(trial,cfg,users,root,source,'cpu',initial)
 assert result['generator_seed']==result['seed']==trial['generator_seed'] and result['training_order_seed']==42
 assert result['test_evaluations']==0 and result['screening_epoch']==2
 last=torch.load(root/'trials'/name/'generator-last.pt',weights_only=False)
 assert last['epoch']==2 and last['updates']==6
assert initials[0]!=initials[1]
report=saved_same_index_oracle(cfg,users,root)
assert report['generator_seeds']==[42,43] and len(report['epochs'])==2
assert (root/'oracle-user-ranks.npz').is_file()
# Corrupting cross-model user pairing or SIDbytes must fail, not silently pool.
p=root/'trials/seed43/validation-epoch-001.npz'
with np.load(p) as f:bad={k:f[k] for k in f.files}
bad['user_ids']=bad['user_ids'][::-1];np.savez_compressed(p,**bad)
try:saved_same_index_oracle(cfg,users,root)
except AssertionError:pass
else:raise AssertionError('mispaired users accepted')
np.save(root/'indices/seed43/semantic-ids.npy',index.sids[::-1])
try:saved_same_index_oracle(cfg,users,root)
except ValueError:pass
else:raise AssertionError('different SID tables accepted')
# Exercise the actual parent plan, isolated worker subprocesses and seed-group
# initialization guard as well; differing seeds must not fail after training.
from unittest.mock import patch
from rhpr.gr.generator_suite import run_generator_suite
from rhpr.gr.data import file_sha256
from rhpr.metrics import validate_metrics
data=root/'resources/data';data.mkdir(parents=True)
(data/'catalog.json').write_text(json.dumps([dict(item_id=i,text=f'Item{i}') for i in range(24)]))
(data/'sequences.json').write_text(json.dumps([dict(user_id=i,train=list(u.train),validation=u.validation,test=20+i) for i,u in enumerate(users)]))
cfg.update(name='tiny-seed-oracle',compare_initialization_by_seed=True,
 execution=dict(worker_devices=['cpu','cpu'],cpu_threads_per_worker=1),
 resources=dict(path=str(root/'resources'),manifest_sha256='fixture',scratch_path=str(root/'scratch')),
 frozen_indices=dict(path='fixture'))
cfg['data']['sha256']={n:file_sha256(data/n) for n in ['catalog.json','sequences.json']}
cfg['evaluation']['paired_same_index_oracle']=True
def prepare(config,ids,output):
 records={}
 for t in config['trials']:
  d=output/'indices'/t['id'];d.mkdir(parents=True)
  np.save(d/'semantic-ids.npy',index.sids)
  (d/'index-provenance.json').write_text(json.dumps(dict(preserve_full_sids=True)))
  records[t['id']]=dict(semantic_ids_sha256=file_sha256(d/'semantic-ids.npy'))
 return records
out=root/'parallel';out.mkdir()
with patch('rhpr.offline_runtime.resolve_dataset_directory',return_value=root/'resources'),patch('rhpr.gr.resources.verify_resources',return_value={}),patch('rhpr.gr.resources.unpack_grid',return_value=Path(source)),patch('rhpr.gr.full_sid_index.prepare_full_sid_indices',side_effect=prepare):
 metrics=validate_metrics(run_generator_suite(cfg,out))
assert metrics['runs']==2 and 'oracle_ndcg10' in metrics
plan=json.loads((out/'workers/plan.json').read_text())
assert len(set(plan['initial_state_sha256_by_trial'].values()))==2
summary=json.loads((out/'suite-results.json').read_text());assert not summary['failures']
assert summary['results']['seed43']['seed']==43
assert summary['results']['seed43']['training_order_seed']==42
'''
        with tempfile.TemporaryDirectory() as directory:
            p=subprocess.run([sys.executable,'-c',script,SOURCE,directory],capture_output=True,text=True,timeout=180)
            self.assertEqual(p.returncode,0,p.stdout+p.stderr)


if __name__=='__main__':unittest.main()
