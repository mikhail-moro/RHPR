"""Catalogue-preserving randomization and observational teacher-forced metrics."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from rhpr.gr.generator_suite import assignments
from rhpr.gr.sid_corruption import corruption_owner,corruption_diagnostics,prepare_corrupted_indices
from rhpr.gr.sid_graph_swap import NeighborGraph

ROOT=Path(__file__).resolve().parents[1]
SOURCE=os.environ.get('RHPR_GRID_SOURCE')
ASSETS=os.environ.get('RHPR_CONTEXT_SCREEN_ASSETS')
SEMANTIC=os.environ.get('RHPR_ALS_SCREEN_ASSETS')


class CorruptionTests(unittest.TestCase):
    def config(self):
        return json.loads((ROOT/'experiments/amazon_beauty_sid_corruption_2t4.json').read_text())

    def test_nested_exact_doses_preserve_paths_and_prefix_occupancies(self):
        n=100
        original=np.array([[i//20,i//5%4,i%5,0] for i in range(n)])
        graph=NeighborGraph(np.arange(n),np.roll(np.arange(n),1),np.ones(n),n)
        previous=set()
        for p in [0,.05,.2,.5,1]:
            owner,selected=corruption_owner(n,p,2753)
            self.assertTrue(previous.issubset(set(selected)));previous=set(selected)
            self.assertEqual(np.count_nonzero(owner!=np.arange(n)),round(n*p))
            np.testing.assert_array_equal(np.sort(owner),np.arange(n))
            np.testing.assert_array_equal(owner,corruption_owner(n,p,2753)[0])
            r=corruption_diagnostics(original,owner,np.arange(1,n+1),graph)
            self.assertTrue(all(x['catalogue_occupancies_exact'] for x in r['levels']))
            if p==0:self.assertEqual(r['levels'][0]['train_target_prefix_total_variation'],0)
        with self.assertRaises(ValueError):corruption_owner(10,.1,1)

    def test_five_fits_match_previous_pure_index_training_protocol(self):
        c=self.config();old=json.loads((ROOT/'experiments/amazon_beauty_sid_context_screen_2t4.json').read_text())
        for k in ['seed','data','generator','quantizer','training','execution','expected_initial_state_sha256','expected_train_cases_sha256','frozen_indices']:
            self.assertEqual(c[k],old[k])
        self.assertEqual([t['corruption_fraction'] for t in c['trials']],[0,.05,.2,.5,1])
        self.assertEqual([len(a['trials']) for a in assignments(c,2)],[3,2])
        self.assertTrue(c['evaluation']['save_teacher_forced_diagnostics'])
        bad=copy.deepcopy(c);bad['trials'][1]['corruption_fraction']=0
        with self.assertRaises(ValueError):assignments(bad,2)

    @unittest.skipUnless(ASSETS and SEMANTIC,'Set both real frozen asset directories')
    def test_real_assets_prepare_exact_nested_maps_and_frequency_diagnostics(self):
        from rhpr.gr.data import load_beauty
        from rhpr.gr.full_sid_index import load_generator_index
        c=self.config();c['frozen_indices']['path']=ASSETS
        c['sid_corruption']['semantic_graph']['path']=SEMANTIC
        _,users,ids,_=load_beauty(Path('/workspace/scratch/gr-beauty-input/data'),c['data'])
        original=np.load(Path(ASSETS)/'original.npy')
        previous=set()
        with tempfile.TemporaryDirectory() as d:
            records=prepare_corrupted_indices(c,ids,users,Path(d))
            for t in c['trials']:
                p=Path(d)/'indices'/t['id'];owner=np.load(p/'owner.npy')
                selected=np.load(p/'selected-items.npy')
                self.assertTrue(previous.issubset(set(selected)));previous=set(selected)
                index=load_generator_index(p,256)
                np.testing.assert_array_equal(index.sids,original[owner])
                self.assertEqual(records[t['id']]['corruption']['changed_items'],round(len(ids)*t['corruption_fraction']))
                self.assertEqual(records[t['id']]['source']['role'],t['role'])

    @unittest.skipUnless(SOURCE,'Set RHPR_GRID_SOURCE for native teacher-forced and training check')
    def test_native_teacher_loss_rng_and_checkpoint_diagnostics(self):
        script=r'''
import json,sys
from pathlib import Path
import numpy as np,torch
from rhpr.gr.grid import build_grid_generator
from rhpr.gr.sid import SidIndex
from rhpr.gr.experiment import _loss,_model_batch
from rhpr.gr.sid_corruption import teacher_forced_diagnostics
from rhpr.gr.generator_suite import run_trial
from rhpr.gr.sid_study import state_digest
torch.set_num_threads(1);root=Path(sys.argv[2]);source=sys.argv[1]
index=SidIndex(np.array([[i//6,0,0] for i in range(24)]),width=8)
gen=dict(d_model=8,d_ff=16,d_kv=4,num_heads=2,encoder_layers=1,decoder_layers=1,dropout=.1,mlp_layers=None,num_user_bins=8,separator_token=False,codebook_size=8,beam_width=2,should_check_prefix=False)
m=build_grid_generator(index.sids,gen,42,'cpu',source);m.eval()
c=np.array([[-1,0],[0,1]]);u=np.array([0,1]);t=np.array([2,3])
rng=torch.get_rng_state().clone();before=state_digest(m.state_dict())
report,arrays=teacher_forced_diagnostics(m,(c,u,t),index,1,'cpu','fp32')
assert torch.equal(rng,torch.get_rng_state()) and state_digest(m.state_dict())==before and not m.training
loss=float(_loss(m,_model_batch(c,u,t,index.sids,'cpu')).detach())
assert abs(loss-report['nll_sum_over_codes'])<2e-6
assert arrays['nll_by_level'].shape==arrays['true_token_rank_by_level'].shape==(2,4)
assert np.all((arrays['true_token_rank_by_level']>=1)&(arrays['true_token_rank_by_level']<=8))
cfg=dict(seed=42,data=dict(history_items=2),quantizer=dict(kind='frozen_full_sids',codebook_size=8),generator=gen,
training=dict(device='cpu',sampling='all_train_prefixes',batch_size=4,physical_batch_size=2,learning_rate=.001,weight_decay=.0001,grad_clip_norm=1.,max_epochs=2,min_epochs=2,patience=2,patience_epochs=2,validation_interval=1,validate_first_epoch=False,evaluation_batch_size=2,max_seconds=120,precision='fp32',evaluation_precision='fp32'),
evaluation=dict(test_after_selection=False,save_validation_predictions=True,save_teacher_forced_diagnostics=True))
class User:
 def __init__(self,i):self.user_id=i;self.train=(i,i+1,i+2,i+3);self.validation=i+4
 @property
 def test(self):raise AssertionError('test read')
d=root/'indices/control';d.mkdir(parents=True);np.save(d/'semantic-ids.npy',index.sids)
(d/'index-provenance.json').write_text(json.dumps(dict(preserve_full_sids=True)))
trial=dict(id='control',quantizer_seed=42,freeze_policy='frozen_full_paths')
result=run_trial(trial,cfg,[User(i) for i in range(3)],root,source,'cpu')
assert result['screening_epoch']==2 and result['test_evaluations']==0
for e in [1,2]:
 d=root/'trials/control'
 with np.load(d/f'teacher-forced-epoch-{e:03d}.npz') as f:assert f['nll_by_level'].shape==(3,4)
 assert len(json.loads((d/f'teacher-forced-epoch-{e:03d}.json').read_text())['nll_per_level'])==4
'''
        with tempfile.TemporaryDirectory() as d:
            r=subprocess.run([sys.executable,'-c',script,SOURCE,d],capture_output=True,text=True,timeout=180)
            self.assertEqual(r.returncode,0,r.stdout+r.stderr)


if __name__=='__main__':unittest.main()
