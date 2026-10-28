"""Shared proposals, independent static scoring, and actual snapshot/worker pipeline."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from rhpr.gr.generator_age import common_pool,budget_prefix,validate_age

ROOT=Path(__file__).resolve().parents[1]
SOURCE=os.environ.get('RHPR_GRID_SOURCE')


class AgeTests(unittest.TestCase):
    def test_balanced_pool_deduplicates_and_keeps_all_source_ranks(self):
        p=lambda a,b:dict(a=a,b=b)
        pools={1:[p(0,1),p(0,2)],5:[p(1,0),p(2,3)],20:[p(3,4),p(4,5)]}
        a=common_pool(pools,4,2759);b=common_pool(dict(reversed(list(pools.items()))),4,2759)
        self.assertEqual(a,b);self.assertEqual(len(a),4)
        self.assertEqual({(p['a'],p['b']) for p in a},{(0,1),(0,2),(2,3),(3,4)})
        self.assertEqual(next(p for p in a if p['a']==0 and p['b']==1)['source_ranks'],{'1':1,'5':1})

    def test_budget_reserves_both_static_and_greedy_complete_batches(self):
        c=np.array([[0],[0],[1],[2],[2]]);t=np.array([1,1,2,3,3]);ex=(c,np.arange(5),t)
        pairs=[dict(a=0,b=1),dict(a=2,b=3)]
        chosen,spent=budget_prefix(pairs,ex,4,2,9)
        self.assertEqual(spent,8);self.assertEqual(len(chosen),1)
        self.assertEqual(chosen[0]['affected_rows'],3)
        self.assertEqual(chosen[0]['scored_rows_per_evaluation'],4)
        self.assertEqual(budget_prefix(pairs,ex,4,2,7),([],0))

    def test_production_is_search_only_with_pinned_late_teacher(self):
        c=json.loads((ROOT/'experiments/amazon_beauty_sid_generator_age_2t4.json').read_text())
        old=json.loads((ROOT/'experiments/amazon_beauty_sid_generator_guided_2t4.json').read_text())
        validate_age(c)
        for key in ['data','generator','training','seed','expected_initial_state_sha256']:
            self.assertEqual(c[key],old[key])
        self.assertEqual(c['age_screen']['checkpoint'],old['generator_guided']['checkpoint'])
        bad=copy.deepcopy(c);bad['age_screen']['search']['semantic_lambda']=.3
        with self.assertRaises(ValueError):validate_age(bad)

    @unittest.skipUnless(SOURCE,'Set RHPR_GRID_SOURCE for actual5epoch snapshots/mining/search workers')
    def test_native_early_snapshots_and_all_three_isolated_age_searches(self):
        script=r'''
import copy,json,sys
from pathlib import Path
from unittest.mock import patch
import numpy as np,torch
from rhpr.gr.data import file_sha256
from rhpr.gr.sid import SidIndex
from rhpr.gr.generator_suite import build_trial_generator
from rhpr.gr.rqvae_experiment import _train_rqvae_generator
from rhpr.gr.generator_age import run_age_screen,compare_teacher
from rhpr.gr.sid_study import state_digest
from rhpr.metrics import validate_metrics
torch.set_num_threads(1);source=Path(sys.argv[1]);root=Path(sys.argv[2])
index=SidIndex(np.array([[i//6,0,0] for i in range(24)]),width=8)
gen=dict(d_model=8,d_ff=16,d_kv=4,num_heads=2,encoder_layers=1,decoder_layers=1,dropout=.1,mlp_layers=None,num_user_bins=8,separator_token=False,codebook_size=8,beam_width=2,should_check_prefix=False)
data=root/'resources/data';data.mkdir(parents=True)
(data/'catalog.json').write_text(json.dumps([dict(item_id=i,text=f'Item{i}') for i in range(24)]))
(data/'sequences.json').write_text(json.dumps([dict(user_id=i,train=[i,i+1,i+2,i+3],validation=i+4,test=i+20) for i in range(3)]))
from rhpr.gr.data import load_beauty
_,users,_,_=load_beauty(data)
c=dict(seed=42,data=dict(history_items=2,sha256={n:file_sha256(data/n) for n in ['catalog.json','sequences.json']}),quantizer=dict(kind='frozen_full_sids',codebook_size=8),generator=gen,
training=dict(device='cpu',cpu_threads=1,sampling='all_train_prefixes',batch_size=4,physical_batch_size=2,learning_rate=.001,weight_decay=.0001,grad_clip_norm=1.,max_epochs=20,min_epochs=20,patience=5,patience_epochs=20,validation_interval=5,validate_first_epoch=False,evaluation_batch_size=2,max_seconds=120,precision='fp32',evaluation_precision='fp32'),
evaluation=dict(test_after_selection=False,save_validation_predictions=False,save_teacher_forced_diagnostics=False),execution=dict(worker_devices=['cpu','cpu'],cpu_threads_per_worker=1),resources=dict(path=str(root/'resources'),manifest_sha256='fixture',scratch_path=str(root/'scratch')),frozen_indices=dict(path='fixture'))
parent=root/'parent';parent.mkdir();model=build_trial_generator(index,c,'cpu',source)
c['expected_initial_state_sha256']=state_digest(dict(model.named_parameters()))
_train_rqvae_generator(model,users,index,c,parent,'cpu')
saved=torch.load(parent/'generator-last.pt',weights_only=False)
torch.save(dict(model=saved['model'],epoch=20,updates=saved['updates'],config=c),parent/'source.pt');del saved
np.save(parent/'sids.npy',index.sids);(parent/'run-manifest.json').write_text(json.dumps(dict(config=c,git_sha='fixture')))
def rec(p):return dict(path=p,sha256=file_sha256(parent/p),size_bytes=(parent/p).stat().st_size)
c['trials']=[dict(id=f'epoch_{e:03d}',epoch=e) for e in [1,5,20]]
spec=dict(batch_size=4,mining_rows=9,seed=2758,minimum_target_count=1,confidence_log_margin=.01,gap_clip=5.,max_proposals=2000,max_scored_rows=10000,max_seconds=120.,min_total_gain=.00001,min_gain_per_affected=.0000001,identity_tolerance=.00001,semantic_lambda=0,precision='fp32')
r=rec('run-manifest.json')
c['age_screen']=dict(epochs=[1,5,20],new_training_epochs=5,pool_size=2000,pool_order_seed=2759,source_training=copy.deepcopy(c['training']),parent=dict(kernel='owner/fixture',path=str(parent),source_code_sha='fixture',files={'run-manifest.json':{k:v for k,v in r.items() if k!='path'}}),checkpoint=rec('source.pt'),sids=rec('sids.npy'),checkpoint_updates=60,search=spec)
from rhpr.gr.generator_suite import training_manifest
c['expected_train_cases_sha256']=training_manifest(users,c)['train_cases_sha256']
def put(config,ids,output):
 for t in config['trials']:
  d=output/'indices'/t['id'];d.mkdir(parents=True);np.save(d/'semantic-ids.npy',index.sids)
  (d/'index-provenance.json').write_text(json.dumps(dict(preserve_full_sids=True)))
out=root/'out';out.mkdir()
# Force a small shared list to exercise actual static/greedy scores and subprocesses;
# real mining is covered separately by test_gr_generator_guided.
pairs=[dict(a=0,b=1),dict(a=2,b=3),dict(a=1,b=4),dict(a=3,b=5)]
with patch('rhpr.offline_runtime.resolve_dataset_directory',return_value=root/'resources'),patch('rhpr.gr.resources.verify_resources',return_value={}),patch('rhpr.gr.resources.unpack_grid',return_value=source),patch('rhpr.gr.full_sid_index.prepare_full_sid_indices',side_effect=put),patch('rhpr.gr.generator_age.mine_pairs',return_value=(pairs,dict(rows=9))):
 result=validate_metrics(run_age_screen(c,out))
assert result['new_training_epochs']==5 and result['downstream_fits']==0 and result['shared_proposals']==4
early=json.loads((out/'early-teacher/training-curves.json').read_text())
late=json.loads((parent/'training-curves.json').read_text())
assert [r['train_loss_sum_over_codes'] for r in early]==[r['train_loss_sum_over_codes'] for r in late[:5]],'snapshotting changed prefix training trajectory'
summary=json.loads((out/'suite-results.json').read_text());assert not summary['failures'] and len(summary['results'])==3
for e,updates in [(1,3),(5,15)]:
 ck=torch.load(out/'early-teacher'/f'generator-epoch-{e:03d}.pt',weights_only=True)
 assert ck['epoch']==e and ck['updates']==updates
for r in summary['results'].values():
 assert r['frozen_weights_and_rng_verified'] and r['cache_audit']['max_row_drift']<=1e-5
 assert r['proposals']==4
traces=[[json.loads(x) for x in (out/'searches'/f'epoch_{e:03d}'/'search-trace.jsonl').read_text().splitlines()] for e in [1,5,20]]
assert all([(r['a'],r['b']) for r in t]==[(r['a'],r['b']) for r in traces[0]] for t in traces)
assert len({r['scored_rows'] for r in summary['results'].values()})==1
# Static probes at every position must match independent original-map evaluation.
from rhpr.gr.generator_guided import LikelihoodSwapSearch,model_scores
from rhpr.gr.data import build_examples
ck=torch.load(out/'early-teacher/generator-epoch-001.pt',weights_only=True)
m=build_trial_generator(index,c,'cpu',source);m.load_state_dict(ck['model']);m.eval();ex=build_examples(users,2,'train')
for row in traces[0]:
 s=LikelihoodSwapSearch(index.sids,ex,lambda ids,sids:model_scores(m,ex,sids,ids,'cpu'),4,min_total_gain=1e-5,min_gain_per_affected=1e-7)
 assert abs(s.try_swap(row['a'],row['b'])['delta_log_likelihood']-row['static_delta'])<1e-10
'''
        with tempfile.TemporaryDirectory() as d:
            r=subprocess.run([sys.executable,'-c',script,SOURCE,d],capture_output=True,text=True,timeout=240)
            self.assertEqual(r.returncode,0,r.stdout+r.stderr)


if __name__=='__main__':unittest.main()
