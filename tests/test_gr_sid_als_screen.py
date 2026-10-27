"""ALS input orientation, guided proposals, matched random and validation-only fits."""
import importlib.util,json,os,subprocess,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from rhpr.gr.sid_graph_swap import NeighborGraph
from rhpr.gr.sid_graph_search import PriorityProposals,randomize_changed_paths,fit_als_signal

ROOT=Path(__file__).resolve().parents[1];SOURCE=os.environ.get('RHPR_GRID_SOURCE')
READY=SOURCE and all(importlib.util.find_spec(x) for x in ['torch','transformers','hydra','lightning'])


class SearchTests(unittest.TestCase):
    def test_guided_partner_occupies_friends_branch_without_swapping_friend(self):
        source=np.arange(8);target=(source+4)%8
        g=NeighborGraph(source,target,np.ones(8),8)
        paths=np.array([[i//4,i%4,0,0] for i in range(8)])
        spec=dict(support_shrinkage=10.,anchor_fraction=1.,uniform_fraction=0.,semantic_partner_pool=2)
        p=PriorityProposals(g,np.eye(8),np.full(8,4),spec,42)
        for _ in range(20):
            a,b=p.pair(paths,[0,1]);friend=target[a]
            self.assertNotEqual(b,friend);self.assertNotEqual(paths[a,0],paths[b,0])
            self.assertEqual(paths[b,0],paths[friend,0])

    def test_random_control_matches_changed_items_and_is_different(self):
        priority=np.array([1,2,3,0,4,5,6,7]);random=randomize_changed_paths(priority,9)
        np.testing.assert_array_equal(np.flatnonzero(random!=np.arange(8)),[0,1,2,3])
        np.testing.assert_array_equal(np.sort(random),np.arange(8));self.assertFalse(np.array_equal(random,priority))

    def test_screen_contract_has_six_fits_and_fixed_epoch10_without_test(self):
        c=json.loads((ROOT/'experiments/amazon_beauty_sid_als_screen_2t4.json').read_text())
        self.assertEqual(len(c['trials']),6);self.assertEqual(c['training']['max_epochs'],10)
        self.assertEqual(c['training']['validation_interval'],5)
        self.assertFalse(c['evaluation']['test_after_selection']);self.assertTrue(c['evaluation']['save_validation_predictions'])
        self.assertEqual({t['expected_variant'] for t in c['trials']},{'als_uniform','als_priority','random_changed_set'})
        self.assertTrue(all(r['epoch']==10 for r in c['controls']['runs'].values()))

    @unittest.skipUnless(importlib.util.find_spec('implicit'),'Requires implicit CPU wheel')
    def test_als_uses_only_binary_train_user_item_matrix(self):
        class User:
            def __init__(self,train):self.train=train
            @property
            def validation(self):raise AssertionError('val consulted')
            @property
            def test(self):raise AssertionError('test consulted')
        spec=dict(factors=3,regularization=.1,alpha=20.,iterations=2,seed=42,cpu_threads=1)
        a,u,f,r=fit_als_signal([User([0,0,1]),User([1,2]),User([2,3])],5,spec)
        self.assertEqual(a.shape,(5,3));self.assertEqual(u.shape,(3,3));self.assertEqual(r['binary_interactions'],6)
        np.testing.assert_array_equal(f,[1,2,2,1,0]);self.assertTrue(np.isfinite(a).all())


@unittest.skipUnless(READY,'Requires pinned GRID and native dependencies')
class NativeTests(unittest.TestCase):
    def test_validation_cadence_preserves_weights_rng_and_screen_never_reads_test(self):
        script=r'''
import json,sys,numpy as np,torch,copy
from pathlib import Path
from unittest.mock import patch
from rhpr.gr.grid import prepare_grid_source
from rhpr.gr.sid import SidIndex
from rhpr.gr.generator_suite import run_trial
root=Path(sys.argv[2]);source=prepare_grid_source(sys.argv[1],root/'grid');torch.set_num_threads(1)
class User:
 def __init__(self,i):self.user_id=i;self.train=tuple(range(i,i+4));self.validation=i+4
 @property
 def test(self):raise AssertionError('TEST DATA WAS READ')
users=[User(i) for i in range(3)];sids=SidIndex(np.c_[np.arange(24)//4,np.zeros((24,2),dtype=int)],width=8).sids
config=dict(seed=42,data=dict(history_items=2),quantizer=dict(kind='frozen_full_sids',codebook_size=8),
 generator=dict(d_model=8,d_ff=16,d_kv=4,num_heads=2,encoder_layers=1,decoder_layers=1,dropout=.1,mlp_layers=None,num_user_bins=8,separator_token=False,codebook_size=8,beam_width=2,should_check_prefix=False),
 training=dict(device='cpu',sampling='all_train_prefixes',batch_size=4,physical_batch_size=2,learning_rate=.001,weight_decay=.0001,grad_clip_norm=1.,max_epochs=2,min_epochs=2,patience=2,patience_epochs=2,validation_interval=2,validate_first_epoch=False,evaluation_batch_size=2,max_seconds=120,precision='fp32',evaluation_precision='fp32'),evaluation=dict(test_after_selection=False,save_validation_predictions=False))
results=[]
for name,cadence,save in [('reference',2,False),('screen',1,True)]:
 output=root/name;directory=output/'indices'/name;directory.mkdir(parents=True);np.save(directory/'semantic-ids.npy',sids)
 (directory/'index-provenance.json').write_text(json.dumps(dict(preserve_full_sids=True)))
 c=copy.deepcopy(config);c['training']['validation_interval']=cadence;c['evaluation']['save_validation_predictions']=save
 trial=dict(id=name,quantizer_seed=42,freeze_policy='frozen_full_paths')
 result=run_trial(trial,c,users,output,source,'cpu')
 assert result['test_evaluations']==0 and result['test']=={} and result['screening_epoch']==2
 d=output/'trials'/name;assert not list(d.glob('test*'))
 if save:
  assert (d/'validation-epoch-001.npz').exists() and (d/'validation-epoch-002.npz').exists()
  curves=json.loads((d/'training-curves.json').read_text());assert result['validation_at_budget']==curves[-1]['validation']
 last=torch.load(d/'generator-last.pt',weights_only=False)
 assert last['epoch']==2 and last['updates']==6 and len(last['sid_matrix_sha256'])==64
 for key in ['python_rng','numpy_rng','scaler','optimizer','torch_rng','cuda_rng']:assert key in last
 results.append(last)
a,b=results
for key,value in a['model'].items():assert torch.equal(value,b['model'][key]),key
assert torch.equal(a['torch_rng'],b['torch_rng']) and a['python_rng']==b['python_rng']
assert a['numpy_rng'][0]==b['numpy_rng'][0] and np.array_equal(a['numpy_rng'][1],b['numpy_rng'][1])
for pid,entry in a['optimizer']['state'].items():
 for key,value in entry.items():
  if torch.is_tensor(value):assert torch.equal(value,b['optimizer']['state'][pid][key])
  else:assert value==b['optimizer']['state'][pid][key]
'''
        with tempfile.TemporaryDirectory() as d:
            r=subprocess.run([sys.executable,'-c',script,SOURCE,d],capture_output=True,text=True,timeout=180)
            self.assertEqual(r.returncode,0,r.stdout+r.stderr)


if __name__=='__main__':unittest.main()
