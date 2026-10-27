"""Fixed-layout counterfactual context lengths must not fit or change targets."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from rhpr.gr.context_ablation import truncate_examples

ROOT=Path(__file__).resolve().parents[1]
SOURCE=os.environ.get('RHPR_GRID_SOURCE')
READY=SOURCE and all(importlib.util.find_spec(n) for n in ['torch','transformers','hydra','lightning'])


class ContextTests(unittest.TestCase):
    def test_truncation_preserves_padding_positions_users_targets_and_source(self):
        contexts=np.array([[-1,-1,-1,1,2,3],[-1,4,5,6,7,8],[1,2,3,4,5,6]])
        users=np.arange(3);targets=np.arange(3)+10;before=contexts.copy()
        result=truncate_examples((contexts,users,targets),3)
        np.testing.assert_array_equal(result[0],np.c_[np.full((3,3),-1),contexts[:,-3:]])
        np.testing.assert_array_equal(contexts,before)
        self.assertIs(result[1],users);self.assertIs(result[2],targets)
        self.assertFalse(np.shares_memory(result[0],contexts))
        np.testing.assert_array_equal(truncate_examples((contexts,users,targets),6)[0],contexts)
        for bad in [True,0,7,3.0]:
            with self.assertRaises(ValueError):truncate_examples((contexts,users,targets),bad)

    def test_registered_run_is_only_two_frozen_epoch60_models(self):
        c=json.loads((ROOT/'experiments/amazon_beauty_sid_context_ablation_2t4.json').read_text())
        parent=json.loads((ROOT/'experiments/amazon_beauty_sid_als_long_2t4.json').read_text())
        self.assertEqual(c['training'],{'device':'cuda'})
        self.assertEqual(c['expected_training'],parent['training'])
        self.assertEqual(c['validation'],dict(batch_size=32,precision='fp32',checkpoint_epoch=60))
        self.assertEqual(c['context_ablation']['lengths'],[3,5,9,20])
        primary=[x for x in c['context_ablation']['cohorts'] if x['primary']]
        self.assertEqual(primary,[dict(id='history20plus',minimum_history=20,lengths=[3,5,9,20],primary=True)])
        self.assertEqual([t['id'] for t in c['trials']],['t5_rqkmeans_original','t5_rqkmeans_uniform'])
        self.assertEqual(c['kaggle']['kernel_sources'],['carasik/rhpr-rqkmeans-als-long'])
        for t in c['trials']:
            for k in ['checkpoint','sids','selection','reference_predictions']:
                self.assertEqual(len(t[k]['sha256']),64)

    def test_robustness_has_ten_retained_models_and_explicit_population_policy(self):
        c=json.loads((ROOT/'experiments/amazon_beauty_sid_context_robustness_2t4.json').read_text())
        self.assertEqual(c['training'],{'device':'cuda'})
        self.assertEqual(len(c['trials']),10)
        epochs=[t.get('checkpoint_epoch',c['validation']['checkpoint_epoch']) for t in c['trials']]
        self.assertEqual(epochs.count(20),4);self.assertEqual(epochs.count(10),6)
        cohort=c['context_ablation']['cohorts'][0]
        self.assertEqual(cohort,dict(id='all_users',minimum_history=3,lengths=[3,5,9,20],allow_short_histories=True,primary=True))
        self.assertEqual(len(c['comparisons']),6)
        for pair in c['comparisons']:
            if pair['epoch']==10:
                self.assertTrue(pair['baseline'].endswith('_random'))
                self.assertEqual(pair['control_kind'],'random_reassignment_not_original')


@unittest.skipUnless(READY,'Requires pinned GRID and native dependencies')
class NativeTests(unittest.TestCase):
    def test_actual_readonly_workers_keep_batches_and_unchanged_inputs_exact(self):
        script=r'''
import json,sys,numpy as np,torch
from pathlib import Path
from unittest.mock import patch
from rhpr.gr.grid import prepare_grid_source,build_grid_generator
from rhpr.gr.data import file_sha256,load_beauty,build_examples
from rhpr.gr.sid import SidIndex
from rhpr.gr.experiment import _evaluate
from rhpr.gr.validation_replay import run_validation_replay
from rhpr.metrics import validate_metrics
root=Path(sys.argv[2]);resources=root/'resources';data=resources/'data';data.mkdir(parents=True)
(data/'catalog.json').write_text(json.dumps([dict(item_id=i,text=f'item{i}') for i in range(24)]))
(data/'sequences.json').write_text(json.dumps([dict(user_id=i,train=list(range(i,i+h)),validation=i+h,test=i+h+1) for i,h in enumerate([3,5,6])]))
spec=dict(history_items=6,sha256={n:file_sha256(data/n) for n in ['catalog.json','sequences.json']})
gen=dict(d_model=8,d_ff=16,d_kv=4,num_heads=2,encoder_layers=1,decoder_layers=1,dropout=.1,mlp_layers=None,num_user_bins=8,separator_token=False,codebook_size=8,beam_width=2,should_check_prefix=False)
source=prepare_grid_source(sys.argv[1],root/'grid');torch.set_num_threads(1)
index=SidIndex(np.c_[np.arange(24)//4,np.zeros((24,2),dtype=int)],width=8)
model=build_grid_generator(index.sids,gen,42,'cpu',source)
_,users,_,_=load_beauty(data);examples=build_examples(users,6,'validation');ref,ids,scores=_evaluate(model,examples,index,32,'cpu','fp32')
parent=root/'parent';parent.mkdir();train_spec=dict(fixture=True)
old=dict(seed=42,data=spec,generator=gen,training=train_spec)
(parent/'run-manifest.json').write_text(json.dumps(dict(git_sha='parent',config=old)))
np.save(parent/'sids.npy',index.sids)
torch.save(dict(model=model.state_dict(),epoch=60,updates=0,config=old),parent/'best.pt')
(parent/'selection.json').write_text(json.dumps(dict(selected_epoch=60,selected_validation=ref)))
np.savez_compressed(parent/'reference.npz',semantic_ids=ids,scores=scores,user_ids=examples[1],target_item_ids=examples[2])
def pin(name):
 p=parent/name;return dict(path=name,sha256=file_sha256(p),size_bytes=p.stat().st_size)
reference_pin=pin('reference.npz');reference_parent=root/'reference_parent';reference_parent.mkdir()
(parent/'reference.npz').rename(reference_parent/'reference.npz')
(reference_parent/'run-manifest.json').write_text(json.dumps(dict(git_sha='reference',config=old)))
refmanifest=reference_parent/'run-manifest.json'
c=dict(seed=42,name='fixture',data=spec,generator=gen,expected_training=dict(global_budget_must_not_be_used=True),training=dict(device='cpu'),
 validation=dict(batch_size=32,precision='fp32',checkpoint_epoch=20),
 context_ablation=dict(lengths=[3,5,6],cohorts=[dict(id='history6plus',minimum_history=6,lengths=[3,5,6],primary=True),dict(id='history5plus',minimum_history=5,lengths=[3,5],primary=False),dict(id='all_users',minimum_history=3,lengths=[3,5,6],allow_short_histories=True,primary=False)]),
 resources=dict(path=str(resources),manifest_sha256='fixture',scratch_path=str(root/'scratch')),
 parents={'p':dict(kernel='owner/fixture',path=str(parent),source_code_sha='parent',files={'run-manifest.json':{k:v for k,v in pin('run-manifest.json').items() if k!='path'}}),
          'reference':dict(kernel='owner/reference',path=str(reference_parent),source_code_sha='reference',files={'run-manifest.json':dict(sha256=file_sha256(refmanifest),size_bytes=refmanifest.stat().st_size)})},
 execution=dict(worker_devices=['cpu','cpu'],cpu_threads_per_worker=1),
 trials=[dict(id=f'model{i}',parent='p',checkpoint_epoch=60,expected_training=train_spec,checkpoint=pin('best.pt'),sids=pin('sids.npy'),selection=pin('selection.json'),reference_predictions=reference_pin,reference_predictions_parent='reference') for i in range(2)])
out=root/'out';out.mkdir()
with patch('rhpr.gr.resources.verify_resources',return_value={}),patch('rhpr.gr.resources.unpack_grid',return_value=Path(sys.argv[1])),patch('rhpr.gr.experiment._train_generator',side_effect=AssertionError('training called')):
 metrics=validate_metrics(run_validation_replay(c,out))
assert metrics['models']==2 and metrics['optimizer_steps']==metrics['test_evaluations']==0 and not metrics['training_performed']
assert metrics['validation_evaluations']==6
for t in c['trials']:
 d=out/'trials'/t['id'];r=json.loads((d/'result.json').read_text())
 assert r['model_unchanged'] and r['epoch']==60 and r['validation_evaluations']==3 and r['parent_full_context_predictions_exact']
 assert [v['unchanged_input_users'] for v in r['unchanged_input_controls']]==[3,1,2]
 assert all(v['changed_predictions_with_unchanged_input']==0 for v in r['unchanged_input_controls'])
 assert [(v['id'],v['users']) for v in r['context_cohorts']]==[('history6plus',1),('history5plus',2),('all_users',3)]
 assert r['context_cohorts'][-1]['allow_short_histories']
 assert r['context_cohorts'][-1]['mean_effective_history_by_length']['6']==14/3
 with np.load(d/'context-user-metrics.npz') as f:
  np.testing.assert_array_equal(f['user_ids'],[0,1,2]);np.testing.assert_array_equal(f['context_lengths'],[3,5,6])
  assert f['rank'].shape==(3,3)
 assert len(list(d.glob('context-L*-predictions.npz')))==3
assert not list(out.rglob('*.pt')) and not list(out.rglob('test*'))
'''
        with tempfile.TemporaryDirectory() as directory:
            r=subprocess.run([sys.executable,'-c',script,SOURCE,directory],capture_output=True,text=True,timeout=180)
            self.assertEqual(r.returncode,0,r.stdout+r.stderr)


if __name__=='__main__':unittest.main()
