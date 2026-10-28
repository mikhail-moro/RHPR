"""Whole-path import and actual TIGER training must retain suffix ownership."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from rhpr.gr.sid import SidIndex

ROOT=Path(__file__).resolve().parents[1]
SOURCE=os.environ.get('RHPR_GRID_SOURCE')
READY=SOURCE and all(importlib.util.find_spec(n) for n in ['torch','hydra','lightning','transformers'])


class FullPathTests(unittest.TestCase):
    def test_suffix_is_preserved_not_reassigned_and_invalid_leaf_sets_are_rejected(self):
        semantic=np.array([[0,1,2],[0,1,2],[1,0,2],[1,0,2]])
        initial=SidIndex(semantic,width=4).sids
        swapped=initial[[1,2,3,0]]
        index=SidIndex.from_full_sids(swapped,width=4)
        np.testing.assert_array_equal(index.sids,swapped)
        self.assertFalse(np.array_equal(index.sids,SidIndex(swapped[:,:3],width=4).sids))
        self.assertEqual(index.sid_to_item[tuple(swapped[0])],0)
        self.assertTrue(index.trie.contains(swapped[0]))
        self.assertFalse(index.sids.flags.writeable)
        self.assertEqual(index.diagnostics()['collision_suffix_order'],'preserved_full_paths')
        for bad in [np.array([[0,0,0,2]]),np.array([[0,0,0,0],[0,0,0,0]]),swapped.astype(float)]:
            with self.assertRaises(ValueError):SidIndex.from_full_sids(bad,width=4)

    def test_three_fits_reuse_exact_original_generator_protocol(self):
        from rhpr.gr.generator_suite import assignments
        c=json.loads((ROOT/'experiments/amazon_beauty_sid_graph_tiger_2t4.json').read_text())
        old=json.loads((ROOT/'experiments/amazon_beauty_field_gradnorm_tiger_2t4.json').read_text())
        for k in ['seed','data','generator','training','execution','expected_initial_state_sha256','expected_train_cases_sha256']:
            self.assertEqual(c[k],old[k])
        self.assertEqual(c['quantizer']['kind'],'frozen_full_sids')
        self.assertEqual([[t['id'] for t in a['trials']] for a in assignments(c,2)],
            [['kmeans_original','kmeans_swapped'],['rqvae_swapped']])
        self.assertEqual(c['controls']['rqvae_original']['epochs'],20)
        self.assertFalse(c['kaggle']['kernel_sources'])
        for trial in c['trials']:
            self.assertEqual(trial['semantic_lambda'],.3 if trial['role']=='swapped' else 0)


@unittest.skipUnless(READY,'Requires pinned GRID and native dependencies')
class NativeTests(unittest.TestCase):
    def test_three_actual_fits_on_frozen_paths_and_owner_identity_guards(self):
        script=r'''
import json,sys,numpy as np
from pathlib import Path
from unittest.mock import patch
from rhpr.gr.data import file_sha256
from rhpr.gr.sid import SidIndex
from rhpr.gr.full_sid_index import prepare_full_sid_indices,load_generator_index
from rhpr.gr.generator_suite import run_generator_suite
from rhpr.metrics import validate_metrics
root=Path(sys.argv[2]);data=root/'resources/data';data.mkdir(parents=True);assets=root/'assets';assets.mkdir()
(data/'catalog.json').write_text(json.dumps([dict(item_id=i,text=f'item{i}') for i in range(24)]))
(data/'sequences.json').write_text(json.dumps([dict(user_id=i,train=[i,i+1,i+2,i+3],validation=i+4,test=i+5) for i in range(3)]))
spec=dict(history_items=2,sha256={n:file_sha256(data/n) for n in ['catalog.json','sequences.json']})
(assets/'catalogue.json').write_text(json.dumps(dict(original_item_ids=list(range(24)))))
semantic=np.array([[i//6,0,0] for i in range(24)])
initial=SidIndex(semantic,width=8).sids;owner=np.arange(24);owner[[0,1,7,8]]=owner[[1,7,8,0]]
np.save(assets/'original.npy',initial);np.save(assets/'swapped.npy',initial[owner])
np.save(assets/'owner.npy',owner);np.save(assets/'identity.npy',np.arange(24))
indices={}
for name,role in [('base','original'),('swap','swapped')]:
 indices[name]=dict(family='fixture',role=role,semantic_lambda=.3 if role=='swapped' else 0,
  sids='swapped.npy' if role=='swapped' else 'original.npy',original_sids='original.npy',owner='owner.npy' if role=='swapped' else 'identity.npy')
files=[dict(path=p.name,sha256=file_sha256(p),size_bytes=p.stat().st_size) for p in assets.iterdir()]
(assets/'resources-manifest.json').write_text(json.dumps(dict(kind='frozen_sid_graph_placement',data_sha256=spec['sha256'],files=files,indices=indices)))
c=dict(seed=42,name='fixture',kind='gr_frozen_sid_generator_suite',data=spec,
 resources=dict(path=str(root/'resources'),manifest_sha256='fixture',scratch_path=str(root/'scratch')),
 frozen_indices=dict(path=str(assets),manifest_sha256=file_sha256(assets/'resources-manifest.json')),
 quantizer=dict(kind='frozen_full_sids',num_hierarchies=3,codebook_size=8),
 generator=dict(d_model=8,d_ff=16,d_kv=4,num_heads=2,encoder_layers=1,decoder_layers=1,dropout=.1,mlp_layers=None,num_user_bins=8,separator_token=False,codebook_size=8,beam_width=2,should_check_prefix=False),
 training=dict(device='cpu',sampling='all_train_prefixes',batch_size=4,physical_batch_size=2,learning_rate=.001,weight_decay=.0001,grad_clip_norm=1.,max_epochs=1,min_epochs=1,patience=1,patience_epochs=1,validation_interval=1,validate_first_epoch=False,evaluation_batch_size=2,max_seconds=120,precision='fp32',evaluation_precision='fp32'),
 execution=dict(worker_devices=['cpu','cpu'],cpu_threads_per_worker=1),
 trials=[dict(id=name,asset_id=asset,family='fixture',role=role,semantic_lambda=.3 if role=='swapped' else 0,quantizer_seed=42,freeze_policy='frozen_full_paths') for name,asset,role in [('control','base','original'),('changed','swap','swapped'),('second_changed','swap','swapped')]])
out=root/'out';out.mkdir()
# Main resource wrapper mocked, but frozen-index byte verification remains real.
with patch('rhpr.gr.resources.verify_resources',return_value={}),patch('rhpr.gr.resources.unpack_grid',return_value=Path(sys.argv[1])),patch('rhpr.gr.ae_rqkmeans.fit_residual_kmeans',side_effect=AssertionError('quantizer fit')):
 metrics=validate_metrics(run_generator_suite(c,out))
assert metrics['runs']==3 and not metrics['quantizer_retrained'] and not metrics['autoencoder_retrained']
for trial in c['trials']:
 directory=out/'indices'/trial['id'];loaded=load_generator_index(directory,8)
 np.testing.assert_array_equal(loaded.sids,initial if trial['role']=='original' else initial[owner])
 result=json.loads((out/'trials'/trial['id']/'result.json').read_text())
 assert result['selection']['epochs']==result['test_evaluations']==1
 assert (out/'trials'/trial['id']/'generator-best.pt').exists()
# Wrong role must fail even when every file hash is valid.
bad={**c,'trials':[{**c['trials'][1],'role':'original'}]}
try:prepare_full_sid_indices(bad,list(range(24)),root/'bad')
except ValueError as e:assert 'identity' in str(e)
else:raise AssertionError('wrong candidate accepted')
'''
        with tempfile.TemporaryDirectory() as directory:
            r=subprocess.run([sys.executable,'-c',script,SOURCE,directory],capture_output=True,text=True,timeout=180)
            self.assertEqual(r.returncode,0,r.stdout+r.stderr)


if __name__=='__main__':unittest.main()
