"""Pinned index reuse, matched TIGER initialization and actual worker execution."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = os.environ.get("RHPR_GRID_SOURCE")
READY = SOURCE and all(importlib.util.find_spec(n) for n in ("torch", "numpy", "hydra", "lightning", "transformers"))


class FrozenSuiteConfigTests(unittest.TestCase):
    def test_generator_and_training_are_the_early_control_with_four_frozen_indices(self):
        from rhpr.gr.generator_suite import assignments
        c = json.loads((ROOT / 'experiments/amazon_beauty_field_warmup_tiger_2t4.json').read_text())
        old = json.loads((ROOT / 'experiments/amazon_beauty_rqvae_tiger_fast_prefix_fp16_20epochs.json').read_text())
        for key in ['seed','data','generator','training','evaluation']:
            self.assertEqual(c[key],old[key])
        self.assertFalse(c['quantizer']['retrain'])
        self.assertEqual({t['freeze_policy'] for t in c['trials']},{'ae','joint'})
        self.assertEqual({t['quantizer_seed'] for t in c['trials']},{42,43})
        self.assertEqual([[t['id'] for t in a['trials']] for a in assignments(c,2)],
            [['ae_s42','joint_s42'],['ae_s43','joint_s43']])
        with self.assertRaisesRegex(RuntimeError,'no hardware fallback'):assignments(c,1)
        c['training']['sampling']='grid_contiguous_windows'
        with self.assertRaisesRegex(ValueError,'all_train_prefixes'):assignments(c,2)

    def test_nested_artifacts_are_byte_pinned_and_cannot_escape_through_symlinks(self):
        from rhpr.gr.generator_suite import verify_file
        from rhpr.gr.data import file_sha256
        with tempfile.TemporaryDirectory() as d:
            root = Path(d);(root/'trials/a').mkdir(parents=True)
            path = root/'trials/a/codes.npy';path.write_bytes(b'pinned')
            record = dict(path='trials/a/codes.npy',sha256=file_sha256(path),size_bytes=6)
            self.assertEqual(verify_file(root,record),path)
            for name in ['../escape','/tmp/escape','..\\escape','']:
                with self.assertRaisesRegex(ValueError,'unsafe'):verify_file(root,{**record,'path':name})
            (root/'alias').symlink_to(root/'trials',target_is_directory=True)
            with self.assertRaisesRegex(ValueError,'unsafe'):verify_file(root,{**record,'path':'alias/a/codes.npy'})
            path.write_bytes(b'edited')
            with self.assertRaisesRegex(ValueError,'digest mismatch'):verify_file(root,record)


@unittest.skipUnless(READY,"Requires pinned GRID and native dependencies")
class NativeFrozenGeneratorTests(unittest.TestCase):
    def test_parent_to_two_workers_runs_four_matched_generators_without_quantizer_fit(self):
        script = r'''
import sys,json,hashlib,numpy as np,torch
from pathlib import Path
from unittest.mock import patch
from rhpr.gr.generator_suite import run_generator_suite,prepare_indices,run_trial
from rhpr.gr.sid import SidIndex
from rhpr.gr.data import file_sha256,load_beauty
from rhpr.metrics import validate_metrics
root=Path(sys.argv[2]);data=root/'resources/data';data.mkdir(parents=True)
(data/'catalog.json').write_text(json.dumps([dict(item_id=i,text=f'item{i}') for i in range(8)]))
(data/'sequences.json').write_text(json.dumps([dict(user_id=i,train=[i,i+1,i+2,i+3],validation=i+4,test=i+5) for i in range(3)]))
parent=root/'parent';parent.mkdir()
data_spec=dict(history_items=2,sha256={n:file_sha256(data/n) for n in ['catalog.json','sequences.json']})
(parent/'run-manifest.json').write_text(json.dumps(dict(git_sha='pinned-parent',config=dict(data=data_spec))))
(parent/'input-provenance.json').write_text(json.dumps(dict(original_item_ids=list(range(8)))))
def record(path):return dict(path=path.relative_to(parent).as_posix(),sha256=file_sha256(path),size_bytes=path.stat().st_size)
trials=[]
for mode in ['ae','joint']:
 for seed in [42,43]:
  name=f'{mode}_s{seed}';directory=parent/'trials'/name;directory.mkdir(parents=True)
  codes=np.zeros((8,3),dtype='int64');codes[:,0]=np.arange(8) if mode=='ae' else np.arange(8)//2
  if seed==43:codes[:,0]=(codes[:,0]+1)%8
  np.save(directory/'semantic-codes.npy',codes)
  diag=SidIndex(codes,width=8).diagnostics()
  q=dict(seed=seed,warmup=dict(freeze_policy=mode,frozen_state_verified=True),config=dict(num_hierarchies=3,codebook_size=8),semantic_code_statistics=diag)
  (directory/'quantizer.json').write_text(json.dumps(q))
  trials.append(dict(id=name,quantizer_seed=seed,freeze_policy=mode,semantic_codes=record(directory/'semantic-codes.npy'),quantizer_report=record(directory/'quantizer.json')))
config=dict(seed=42,name='fixture',kind='gr_frozen_sid_generator_suite',data=data_spec,
 resources=dict(path=str(root/'resources'),manifest_sha256='fixture',scratch_path=str(root/'scratch')),
 parent=dict(kernel='owner/fixture',path=str(parent),source_code_sha='pinned-parent',files={n:{k:v for k,v in record(parent/n).items() if k!='path'} for n in ['run-manifest.json','input-provenance.json']}),
 quantizer=dict(kind='frozen_field_rqvae_index',num_hierarchies=3,codebook_size=8,retrain=False),
 generator=dict(d_model=8,d_ff=16,d_kv=4,num_heads=2,encoder_layers=1,decoder_layers=1,dropout=.1,mlp_layers=None,num_user_bins=8,separator_token=False,codebook_size=8,beam_width=2,should_check_prefix=False),
 training=dict(device='cpu',sampling='all_train_prefixes',batch_size=4,physical_batch_size=2,learning_rate=.001,weight_decay=.0001,grad_clip_norm=1.,max_epochs=2,min_epochs=2,patience=2,patience_epochs=2,validation_interval=2,validate_first_epoch=False,evaluation_batch_size=2,max_seconds=120,precision='fp32',evaluation_precision='fp32'),
 execution=dict(worker_devices=['cpu','cpu'],cpu_threads_per_worker=1),trials=trials)
output=root/'output';output.mkdir()
with patch('rhpr.offline_runtime.resolve_dataset_directory',return_value=root/'resources'),patch('rhpr.gr.resources.verify_resources',return_value={}),patch('rhpr.gr.resources.unpack_grid',return_value=Path(sys.argv[1])):
 metrics=validate_metrics(run_generator_suite(config,output))
assert metrics['runs']==4 and not metrics['quantizer_retrained']
results=json.loads((output/'suite-results.json').read_text())
assert not results['failures'] and len(results['results'])==4
initial=json.loads((output/'initialization.json').read_text())['initial_state_sha256']
assert {r['initial_state_sha256'] for r in results['results'].values()}=={initial}
stream=json.loads((output/'shared-training-stream.json').read_text())
assert stream['examples_per_epoch']==9 and len(stream['epoch_order_sha256'])==2
for name,result in results['results'].items():
 assert result['selection']['epochs']==2 and result['selection']['model_examples_consumed']==18
 assert result['selection']['validation_checks']==1 and result['test_evaluations']==1
 directory=output/'trials'/name
 curves=json.loads((directory/'training-curves.json').read_text())
 assert [r['epoch'] for r in curves if 'validation' in r]==[2]
 assert all(r['train_examples']==9 for r in curves)
 assert (directory/'generator-best.pt').exists() and (directory/'generator-last.pt').exists()
 with np.load(directory/'test-predictions.npz') as p:
  assert p['semantic_ids'].shape==(3,2,4)
  np.testing.assert_array_equal(p['target_item_ids'],[5,6,7])
 s=np.load(output/'indices'/name/'semantic-ids.npy');assert s.shape==(8,4)
assert not list(output.rglob('quantizer.pt')) and not list(output.rglob('autoencoder.pt'))
# Frozen rows/order mismatch and tampered bytes must fail before training.
try:prepare_indices(config,parent,list(range(7,-1,-1)),root/'bad')
except ValueError as e:assert 'catalogue/order' in str(e)
else:raise AssertionError('order drift accepted')
path=parent/config['trials'][0]['semantic_codes']['path'];path.write_bytes(b'changed')
try:prepare_indices(config,parent,list(range(8)),root/'tampered')
except ValueError as e:assert 'digest mismatch' in str(e)
else:raise AssertionError('tampered index accepted')
# Incorrect neural-init control is rejected before calling the trainer.
_,users,_,_=load_beauty(data,data_spec)
source=output/'workers/plan.json';source=json.loads(source.read_text())['source']
with patch('rhpr.gr.rqvae_experiment._train_rqvae_generator',side_effect=AssertionError('trainer called')):
 try:run_trial(config['trials'][0],config,users,output,source,'cpu','bad-hash')
 except ValueError as e:assert 'initialization' in str(e)
 else:raise AssertionError('mismatched init accepted')
'''
        with tempfile.TemporaryDirectory() as directory:
            result=subprocess.run([sys.executable,'-c',script,SOURCE,directory],capture_output=True,text=True,timeout=180)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)


if __name__=='__main__':unittest.main()
