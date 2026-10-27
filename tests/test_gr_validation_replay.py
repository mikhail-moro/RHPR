"""Read-only validation replay must restore exact weights/paths and save users."""
import importlib.util,json,os,subprocess,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];SOURCE=os.environ.get('RHPR_GRID_SOURCE')
READY=SOURCE and all(importlib.util.find_spec(x) for x in ['numpy','torch','transformers','hydra','lightning'])


class ContractTests(unittest.TestCase):
    def test_replay_has_only_selected_checkpoints_and_no_training_budget(self):
        c=json.loads((ROOT/'experiments/amazon_beauty_sid_validation_replay_2t4.json').read_text())
        self.assertEqual(c['training'],{'device':'cuda'})
        self.assertEqual(c['validation'],dict(batch_size=32,precision='fp32',checkpoint_epoch=20))
        self.assertEqual(len(c['trials']),4)
        self.assertEqual(c['expected_training']['max_epochs'],20)
        self.assertEqual(len(c['kaggle']['kernel_sources']),2)
        for t in c['trials']:
            for field in ['checkpoint','sids','selection']:
                self.assertEqual(len(t[field]['sha256']),64)


@unittest.skipUnless(READY,'Requires pinned GRID and native dependencies')
class NativeTests(unittest.TestCase):
    def test_two_real_readonly_workers_save_exact_user_metrics_without_training_or_test(self):
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
(data/'sequences.json').write_text(json.dumps([dict(user_id=i,train=[i,i+1,i+2,i+3],validation=i+4,test=i+5) for i in range(3)]))
spec=dict(history_items=2,sha256={n:file_sha256(data/n) for n in ['catalog.json','sequences.json']})
gen=dict(d_model=8,d_ff=16,d_kv=4,num_heads=2,encoder_layers=1,decoder_layers=1,dropout=.1,mlp_layers=None,num_user_bins=8,separator_token=False,codebook_size=8,beam_width=2,should_check_prefix=False)
source=prepare_grid_source(sys.argv[1],root/'grid');torch.set_num_threads(1)
index=SidIndex(np.c_[np.arange(24)//4,np.zeros((24,2),dtype=int)],width=8)
model=build_grid_generator(index.sids,gen,42,'cpu',source)
_,users,_,_=load_beauty(data);ref,_,_=_evaluate(model,build_examples(users,2,'validation'),index,32,'cpu','fp32')
parent=root/'parent';parent.mkdir();train_spec=dict(fixture=True)
old=dict(seed=42,data=spec,generator=gen,training=train_spec)
(parent/'run-manifest.json').write_text(json.dumps(dict(git_sha='parent',config=old)))
np.save(parent/'sids.npy',index.sids)
torch.save(dict(model=model.state_dict(),epoch=20,updates=10,config=old),parent/'best.pt')
(parent/'selection.json').write_text(json.dumps(dict(selected_validation=ref)))
def pin(name):
 p=parent/name;return dict(path=name,sha256=file_sha256(p),size_bytes=p.stat().st_size)
c=dict(seed=42,name='fixture',data=spec,generator=gen,expected_training=train_spec,training=dict(device='cpu'),
 validation=dict(batch_size=32,precision='fp32',checkpoint_epoch=20),
 resources=dict(path=str(resources),manifest_sha256='fixture',scratch_path=str(root/'scratch')),
 parents={'p':dict(kernel='owner/fixture',path=str(parent),source_code_sha='parent',files={'run-manifest.json':{k:v for k,v in pin('run-manifest.json').items() if k!='path'}})},
 execution=dict(worker_devices=['cpu','cpu'],cpu_threads_per_worker=1),
 trials=[dict(id=f'model{i}',parent='p',checkpoint=pin('best.pt'),sids=pin('sids.npy'),selection=pin('selection.json')) for i in range(2)])
out=root/'out';out.mkdir()
with patch('rhpr.gr.resources.verify_resources',return_value={}),patch('rhpr.gr.resources.unpack_grid',return_value=Path(sys.argv[1])),patch('rhpr.gr.experiment._train_generator',side_effect=AssertionError('training called')):
 metrics=validate_metrics(run_validation_replay(c,out))
assert metrics['models']==2 and metrics['optimizer_steps']==metrics['test_evaluations']==0 and not metrics['training_performed']
for t in c['trials']:
 d=out/'trials'/t['id'];r=json.loads((d/'result.json').read_text());assert r['model_unchanged'] and r['validation_evaluations']==1
 assert all(abs(r['replay_delta'][k])<1e-12 for k in ['ndcg@5','ndcg@10','recall@5','recall@10'])
 with np.load(d/'validation-user-metrics.npz') as f:
  assert len(f['user_ids'])==3 and abs(float(f['ndcg@10'].mean())-ref['ndcg@10'])<1e-12
assert not list(out.rglob('*.pt')) and not list(out.rglob('test-predictions.npz'))
'''
        with tempfile.TemporaryDirectory() as d:
            r=subprocess.run([sys.executable,'-c',script,SOURCE,d],capture_output=True,text=True,timeout=180)
            self.assertEqual(r.returncode,0,r.stdout+r.stderr)


if __name__=='__main__':unittest.main()
