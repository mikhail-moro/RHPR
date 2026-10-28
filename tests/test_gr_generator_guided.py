"""Local likelihood deltas include history effects and preserve the batch grid."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from rhpr.gr.generator_guided import LikelihoodSwapSearch,incidence
from rhpr.gr.generator_suite import assignments

ROOT=Path(__file__).resolve().parents[1]
SOURCE=os.environ.get('RHPR_GRID_SOURCE')


class GuidedTests(unittest.TestCase):
    def test_target_improvement_can_be_outweighed_by_other_targets_history_effects(self):
        c=np.array([[2],[1],[1],[1]]);u=np.arange(4);t=np.array([0,2,2,2]);sids=np.arange(3)[:,None]
        def score(ids,table):return -(table[t[ids],0]-table[c[ids,0],0])**2
        search=LikelihoodSwapSearch(sids,(c,u,t),score,2)
        self.assertEqual(search.incidence[1].tolist(),[1,2,3])
        before=search.cache.copy();owner=search.owner.copy();r=search.try_swap(0,1)
        self.assertFalse(r['accepted']);self.assertEqual(r['delta_log_likelihood'],-6)
        self.assertEqual(r['affected_rows'],4)
        np.testing.assert_array_equal(search.cache,before);np.testing.assert_array_equal(search.owner,owner)
        self.assertEqual(search.audit(1e-12)['max_row_drift'],0)

    def test_accepted_cache_matches_full_score_with_stable_batch_shapes(self):
        c=np.array([[2],[2],[0]]);u=np.arange(3);t=np.arange(3);table=np.array([[1],[0],[2]])
        def score(ids,sids):return -(sids[t[ids],0]-u[ids])**2-.01*sids[c[ids,0],0]+len(ids)*.1
        search=LikelihoodSwapSearch(table,(c,u,t),score,2)
        r=search.try_swap(0,1);self.assertTrue(r['accepted'])
        self.assertAlmostEqual(r['delta_log_likelihood'],2.01)
        self.assertEqual(search.owner.tolist(),[1,0,2]);self.assertFalse(search.try_swap(0,1)['accepted'])
        self.assertFalse(search.try_swap(0,0)['accepted']);search.audit(1e-12)
        with self.assertRaises(FloatingPointError):
            LikelihoodSwapSearch(table,(c,u,t),lambda ids,sids:np.full(len(ids),np.nan),2)

    def test_fresh_control_and_candidate_keep_existing_training_and_pinned_source(self):
        c=json.loads((ROOT/'experiments/amazon_beauty_sid_generator_guided_2t4.json').read_text())
        old=json.loads((ROOT/'experiments/amazon_beauty_sid_context_screen_2t4.json').read_text())
        for key in ['seed','data','generator','quantizer','training','execution','expected_initial_state_sha256','expected_train_cases_sha256']:
            self.assertEqual(c[key],old[key])
        self.assertEqual([len(a['trials']) for a in assignments(c,2)],[1,1])
        self.assertEqual(c['kaggle']['kernel_sources'],['carasik/rhpr-sid-corruption-screen'])
        self.assertEqual(c['generator_guided']['checkpoint_epoch'],20)
        self.assertEqual(c['generator_guided']['semantic_lambda'],0)
        self.assertEqual(c['generator_guided']['max_proposals'],2000)
        bad=copy.deepcopy(c);bad['generator_guided']['semantic_lambda']=.3
        with self.assertRaises(ValueError):assignments(bad,2)

    @unittest.skipUnless(SOURCE,'Set RHPR_GRID_SOURCE for actual model scoring/search/fits')
    def test_native_frozen_search_then_two_fresh_isolated_worker_fits(self):
        script=r'''
import json,sys,hashlib
from pathlib import Path
from unittest.mock import patch
import numpy as np,torch
from rhpr.gr.sid import SidIndex
from rhpr.gr.grid import build_grid_generator
from rhpr.gr.generator_suite import run_trial,run_generator_suite
from rhpr.gr.generator_guided import LikelihoodSwapSearch,model_scores,mine_pairs
from rhpr.gr.experiment import _loss,_model_batch
from rhpr.gr.data import file_sha256,build_examples
from rhpr.gr.sid_study import state_digest
from rhpr.metrics import validate_metrics
torch.set_num_threads(1);root=Path(sys.argv[2]);source=Path(sys.argv[1])
index=SidIndex(np.array([[i//6,0,0] for i in range(24)]),width=8)
gen=dict(d_model=8,d_ff=16,d_kv=4,num_heads=2,encoder_layers=1,decoder_layers=1,dropout=.1,mlp_layers=None,num_user_bins=8,separator_token=False,codebook_size=8,beam_width=2,should_check_prefix=False)
class User:
 def __init__(self,i):self.user_id=i;self.train=(i,i+1,i+2,i+3);self.validation=i+4
 @property
 def test(self):raise AssertionError('test read')
users=[User(i) for i in range(3)];examples=build_examples(users,2,'train')
data=root/'resources/data';data.mkdir(parents=True)
(data/'catalog.json').write_text(json.dumps([dict(item_id=i,text=f'Item{i}') for i in range(24)]))
(data/'sequences.json').write_text(json.dumps([dict(user_id=i,train=list(u.train),validation=u.validation,test=20+i) for i,u in enumerate(users)]))
cfg=dict(seed=42,data=dict(history_items=2,sha256={n:file_sha256(data/n) for n in ['catalog.json','sequences.json']}),quantizer=dict(kind='frozen_full_sids',codebook_size=8),generator=gen,
training=dict(device='cpu',sampling='all_train_prefixes',batch_size=4,physical_batch_size=2,learning_rate=.001,weight_decay=.0001,grad_clip_norm=1.,max_epochs=1,min_epochs=1,patience=1,patience_epochs=1,validation_interval=1,validate_first_epoch=False,evaluation_batch_size=2,max_seconds=120,precision='fp32',evaluation_precision='fp32'),
evaluation=dict(test_after_selection=False,save_validation_predictions=True),
execution=dict(worker_devices=['cpu','cpu'],cpu_threads_per_worker=1),resources=dict(path=str(root/'resources'),manifest_sha256='fixture',scratch_path=str(root/'scratch')),frozen_indices=dict(path='fixture'))
parent=root/'parent';parent.mkdir()
def put_indices(config,ids,output):
 records={}
 for trial in config['trials']:
  d=output/'indices'/trial['id'];d.mkdir(parents=True)
  np.save(d/'semantic-ids.npy',index.sids);np.save(d/'owner.npy',np.arange(24))
  record=dict(preserve_full_sids=True,source=dict(role='original'),semantic_ids_sha256=file_sha256(d/'semantic-ids.npy'))
  (d/'index-provenance.json').write_text(json.dumps(record));records[trial['id']]=record
 (output/'sid-catalogue.json').write_text(json.dumps(dict(indices=records,original_item_ids=list(range(24)))))
 return records
cfg['trials']=[dict(id='teacher',quantizer_seed=42,freeze_policy='frozen_full_paths')]
put_indices(cfg,list(range(24)),parent)
result=run_trial(cfg['trials'][0],cfg,users,parent,source,'cpu')
(parent/'run-manifest.json').write_text(json.dumps(dict(git_sha='fixture',config=cfg)))
assert result['selection']['selected_epoch']==1
def rec(name):
 p=parent/name;return dict(path=name,sha256=file_sha256(p),size_bytes=p.stat().st_size)
r=rec('run-manifest.json')
spec=dict(control_trial='original',candidate_trial='guided',parent=dict(kernel='owner/fixture',path=str(parent),source_code_sha='fixture',files={'run-manifest.json':{k:v for k,v in r.items() if k!='path'}}),
checkpoint=rec('trials/teacher/generator-best.pt'),sids=rec('indices/teacher/semantic-ids.npy'),source_result=rec('trials/teacher/result.json'),source_trial='teacher',checkpoint_epoch=1,checkpoint_updates=3,
precision='fp32',batch_size=4,mining_rows=9,seed=7,minimum_target_count=1,confidence_log_margin=.01,gap_clip=5.,max_proposals=276,max_accepted=1,max_scored_rows=10000,max_seconds=120.,min_total_gain=1e-5,min_gain_per_affected=1e-7,identity_tolerance=1e-5,semantic_lambda=0)
m=build_grid_generator(index.sids,gen,42,'cpu',source);m.load_state_dict(torch.load(parent/'trials/teacher/generator-best.pt',weights_only=True)['model']);m.eval()
before=state_digest(m.state_dict());rng=torch.get_rng_state().clone()
score=lambda ids,sids:model_scores(m,examples,sids,ids,'cpu')
s=LikelihoodSwapSearch(index.sids,examples,score,4,min_total_gain=1e-5,min_gain_per_affected=1e-7)
ids=np.arange(4);native=float(_loss(m,_model_batch(examples[0][ids],examples[1][ids],examples[2][ids],index.sids,'cpu')).detach())
assert abs(native+s.cache[:4].mean())<2e-6
mine=root/'mining';mine.mkdir();pairs,report=mine_pairs(m,examples,index.sids,s.cache,spec,'cpu',8,mine)
assert report['rows']==9 and m.top_k_for_generation==2
assert state_digest(m.state_dict())==before and torch.equal(rng,torch.get_rng_state())
for a in range(24):
 for b in range(a+1,24):
  s.try_swap(a,b)
  if s.accepted:break
 if s.accepted:break
assert s.accepted,'fixture has no improving swap'
audit=s.audit(1e-5);assert audit['full_rescore_delta']>0
cfg['trials']=[dict(id=name,quantizer_seed=42,freeze_policy='frozen_full_paths') for name in ['original','guided']]
cfg['generator_guided']=spec;out=root/'out';out.mkdir()
proposals=[dict(a=a,b=b,mistakes=1,priority=1.) for a in range(24) for b in range(a+1,24)]
with patch('rhpr.offline_runtime.resolve_dataset_directory',return_value=root/'resources'),patch('rhpr.gr.resources.verify_resources',return_value={}),patch('rhpr.gr.resources.unpack_grid',return_value=source),patch('rhpr.gr.generator_guided.prepare_originals',side_effect=put_indices),patch('rhpr.gr.generator_guided.mine_pairs',return_value=(proposals,dict(rows=9,selected_pairs=len(proposals)))):
 metrics=validate_metrics(run_generator_suite(cfg,out))
assert metrics['runs']==2 and metrics['guided_accepted']==1
report=json.loads((out/'generator-guided/optimization.json').read_text())
assert report['frozen_weights_and_rng_verified'] and report['cache_audit']['max_row_drift']<1e-5
summary=json.loads((out/'suite-results.json').read_text());assert not summary['failures']
assert len({r['initial_state_sha256'] for r in summary['results'].values()})==1
assert all(r['selection']['epochs']==1 and r['test_evaluations']==0 for r in summary['results'].values())
empty=root/'no-pairs';empty.mkdir()
with patch('rhpr.offline_runtime.resolve_dataset_directory',return_value=root/'resources'),patch('rhpr.gr.resources.verify_resources',return_value={}),patch('rhpr.gr.resources.unpack_grid',return_value=source),patch('rhpr.gr.generator_guided.prepare_originals',side_effect=put_indices),patch('rhpr.gr.generator_guided.mine_pairs',return_value=([],dict(rows=9,selected_pairs=0))):
 skipped=validate_metrics(run_generator_suite(cfg,empty))
assert skipped['runs']==0 and skipped['training_skipped_no_swaps'] and not (empty/'trials').exists()
'''
        with tempfile.TemporaryDirectory() as directory:
            r=subprocess.run([sys.executable,'-c',script,SOURCE,directory],capture_output=True,text=True,timeout=180)
            self.assertEqual(r.returncode,0,r.stdout+r.stderr)


if __name__=='__main__':unittest.main()
