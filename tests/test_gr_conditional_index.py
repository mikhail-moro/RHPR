"""Causal per-prefix routing, branch-aware decoding and real GRID integration."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from rhpr.gr.conditional_index import ConditionalSidIndex
from rhpr.gr.data import build_examples
from rhpr.gr.generator_suite import assignments, training_manifest
from rhpr.gr.sid import SidIndex

ROOT = Path(__file__).resolve().parents[1]
SOURCE = os.environ.get('RHPR_GRID_SOURCE')
NATIVE = SOURCE and all(importlib.util.find_spec(k) for k in ['torch', 'transformers', 'hydra', 'lightning'])


def fixture(threshold=2):
    original = SidIndex(np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]]), width=4)
    changed = SidIndex.from_full_sids(original.sids[[1, 0, 2, 3]], width=4)
    return ConditionalSidIndex(original, changed, threshold)


class RoutingTests(unittest.TestCase):
    def test_gate_uses_available_prefix_not_final_user_history(self):
        class User:
            user_id = 7
            train = (0, 1, 2, 3)
            @property
            def validation(self): raise AssertionError('validation read')
            @property
            def test(self): raise AssertionError('test read')
        contexts, _, _ = build_examples([User()], 2, 'train')
        np.testing.assert_array_equal(fixture().branches(contexts), [0, 1, 1])
        config = dict(seed=42, data=dict(history_items=2), training=dict(max_epochs=2),
                      index_gate=dict(threshold=2))
        stream = training_manifest([User()], config)
        self.assertEqual(stream['index_gate']['short_examples'], 1)
        self.assertEqual(stream['index_gate']['long_examples'], 2)
        self.assertFalse(fixture().sids.flags.writeable)

    def test_exact_item_decoding_depends_on_selected_mapping(self):
        index = fixture()
        contexts = np.array([[-1, 2], [2, 3]])
        short, long = index.indices
        predictions = np.array([[short.sids[0], short.sids[0]], [short.sids[0], long.sids[0]]])
        metrics = index.evaluate(predictions, np.array([0, 0]), contexts)
        self.assertAlmostEqual(metrics['ndcg@10'], (1 + 1/np.log2(3))/2)
        self.assertEqual(metrics['recall@5'], 1.)
        self.assertEqual(metrics['duplicate_predictions'], 1)
        self.assertEqual(metrics['invalid_predictions'], 0)
        self.assertEqual(metrics['gate_short_examples'], 1)
        self.assertEqual(metrics['gate_long_examples'], 1)
        metadata = index.prediction_metadata(contexts)
        np.testing.assert_array_equal(metadata['branch_ids'], [0, 1])
        np.testing.assert_array_equal(metadata['observed_context_lengths'], [1, 2])

    def test_incompatible_paths_and_invalid_prefixes_fail(self):
        index = fixture()
        other = SidIndex(np.array([[0, 1, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]]), width=4)
        with self.assertRaises(ValueError): ConditionalSidIndex(index.indices[0], other, 2)
        for c in [np.array([[0]]), np.array([[-2, 0]]), np.array([[0, 4]])]:
            with self.assertRaises(ValueError): index.branches(c)

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'Requires native torch')
    def test_entire_context_and_target_use_same_branch_with_pads(self):
        index = fixture()
        c = np.array([[-1, 0], [0, 1]])
        batch = index.model_batch(c, np.array([3, 4]), np.array([0, 0]), 'cpu')
        np.testing.assert_array_equal(batch['input_ids'][:, 0], [0, 1])
        np.testing.assert_array_equal(batch['input_ids'][0, 1:5], [-1]*4)
        np.testing.assert_array_equal(batch['attention_mask_encoder'][0], [1, 0, 0, 0, 0, 1, 1, 1, 1])
        np.testing.assert_array_equal(batch['input_ids'][1, 1:], index.indices[1].sids[[0, 1]].ravel())
        np.testing.assert_array_equal(batch['future_ids'], index.sids[[0, 1], [0, 0]])

    def test_launch_has_two_same_feature_twenty_epoch_controls(self):
        c = json.loads((ROOT/'experiments/amazon_beauty_sid_gated_index_2t4.json').read_text())
        old = json.loads((ROOT/'experiments/amazon_beauty_sid_context_screen_2t4.json').read_text())
        for key in ['seed', 'data', 'generator', 'quantizer', 'execution', 'frozen_indices',
                    'expected_train_cases_sha256', 'training']:
            self.assertEqual(c[key], old[key])
        self.assertEqual(c['index_gate']['threshold'], 10)
        self.assertEqual(c['index_gate']['token'], 'encoder_prefix')
        self.assertTrue(c['trials'][0]['use_original_paths'])
        self.assertEqual(c['trials'][1]['expected_variant'], 'als_uniform')
        self.assertEqual([[t['id'] for t in a['trials']] for a in assignments(c, 2)],
                         [['original_with_branch'], ['gated_original_als']])
        self.assertFalse(c['evaluation']['test_after_selection'])
        self.assertEqual(c['training']['max_epochs'], 20)
        self.assertEqual(c['training']['validation_interval'], 5)


@unittest.skipUnless(NATIVE, 'Requires pinned GRID source and native dependencies')
class NativeTests(unittest.TestCase):
    def test_real_prefix_embedding_teacher_loss_cached_beams_training_and_restore(self):
        script = r'''
import copy,json,sys,numpy as np,torch
from pathlib import Path
from rhpr.gr.grid import build_grid_generator
from rhpr.gr.conditional_index import ConditionalSidIndex,add_branch_token
from rhpr.gr.sid import SidIndex
from rhpr.gr.experiment import _model_batch,_loss,_evaluate
from rhpr.gr.generator_suite import run_trial,build_trial_generator
from rhpr.gr.sid_study import state_digest
torch.set_num_threads(1);root=Path(sys.argv[2]);source=sys.argv[1]
original=SidIndex(np.array([[i//6,0,0] for i in range(24)]),width=8)
owner=np.arange(24);owner[[0,1,7,8]]=owner[[1,7,8,0]]
changed=SidIndex.from_full_sids(original.sids[owner],width=8)
index=ConditionalSidIndex(original,changed,2);control=ConditionalSidIndex(original,original,2)
gen=dict(d_model=8,d_ff=16,d_kv=4,num_heads=2,encoder_layers=1,decoder_layers=1,
 dropout=.1,mlp_layers=None,num_user_bins=8,separator_token=False,codebook_size=8,beam_width=2,should_check_prefix=False)
c=np.array([[-1,0],[0,1]]);uids=np.array([0,1]);targets=np.array([0,0])
model=build_grid_generator(original.sids,gen,42,'cpu',source);model.eval()
captured=[]
def record(module,args,kwargs):captured.append((kwargs['sequence_embedding'].detach().clone(),kwargs['attention_mask'].detach().clone()))
hook=model.encoder.register_forward_pre_hook(record,with_kwargs=True)
batch=_model_batch(c,uids,targets,original.sids,'cpu')
model.encoder_forward_pass(batch['attention_mask_encoder'],batch['input_ids'],batch['user_id'])
add_branch_token(model);routed=control.model_batch(c,uids,targets,'cpu')
model.encoder_forward_pass(routed['attention_mask_encoder'],routed['input_ids'],routed['user_id'])
hook.remove();a,b=captured
assert torch.equal(b[0][:,1:],a[0]) and torch.equal(b[1][:,1:],a[1]),'native hierarchy/user preparation changed'
assert torch.equal(b[0][:,0],model.index_branch_embedding(torch.tensor([0,1])))
model.train();model.zero_grad();loss=_loss(model,index.model_batch(c,uids,targets,'cpu'));assert torch.isfinite(loss)
loss.backward();assert torch.all(model.index_branch_embedding.weight.grad.norm(dim=1)>0)
score,beams,prob=_evaluate(model,(c,uids,targets),index,2,'cpu')
assert beams.shape==(2,2,4) and np.isfinite(prob).all()
assert score['gate_short_examples']==score['gate_long_examples']==1
cfg=dict(seed=42,data=dict(history_items=2),quantizer=dict(kind='frozen_full_sids',codebook_size=8),
 generator=gen,index_gate=dict(threshold=2,short_trial='original_with_branch',token='encoder_prefix'),
 training=dict(device='cpu',sampling='all_train_prefixes',batch_size=4,physical_batch_size=2,
 learning_rate=.001,weight_decay=.0001,grad_clip_norm=1.,max_epochs=2,min_epochs=2,
 patience=2,patience_epochs=2,validation_interval=1,validate_first_epoch=False,
 evaluation_batch_size=2,max_seconds=120,precision='fp32',evaluation_precision='fp32'),
 evaluation=dict(test_after_selection=False,save_validation_predictions=True))
class User:
 def __init__(self,i):self.user_id=i;self.train=(i,i+1,i+2,i+3);self.validation=i+4
 @property
 def test(self):raise AssertionError('test read')
users=[User(i) for i in range(3)]
for name,paths in [('original_with_branch',original.sids),('gated_original_als',changed.sids)]:
 d=root/'indices'/name;d.mkdir(parents=True);np.save(d/'semantic-ids.npy',paths)
 (d/'index-provenance.json').write_text(json.dumps(dict(preserve_full_sids=True)))
a=build_trial_generator(control,cfg,'cpu',source);b=build_trial_generator(index,cfg,'cpu',source)
assert state_digest(dict(a.named_parameters()))==state_digest(dict(b.named_parameters()))
initial=state_digest(dict(a.named_parameters()))
for name,idx in [('original_with_branch',control),('gated_original_als',index)]:
 trial=dict(id=name,quantizer_seed=42,freeze_policy='frozen_full_paths')
 result=run_trial(trial,cfg,users,root,source,'cpu',initial)
 assert result['test_evaluations']==0 and result['screening_epoch']==2
 d=root/'trials'/name
 for epoch in [1,2]:
  with np.load(d/f'validation-epoch-{epoch:03d}.npz') as f:assert np.array_equal(f['branch_ids'],[1,1,1])
 last=torch.load(d/'generator-last.pt',weights_only=False)
 assert last['sid_matrix_sha256']==idx.provenance()['sid_matrix_sha256'] and last['epoch']==2
 restored=build_trial_generator(idx,cfg,'cpu',source);restored.load_state_dict(last['model'])
 assert torch.equal(restored.index_branch_embedding.weight,last['model']['index_branch_embedding.weight'])
 assert last['updates']==6 and len(last['optimizer']['state'])>0
 assert last['config']['index_gate']==cfg['index_gate']
'''
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([sys.executable, '-c', script, SOURCE, directory],
                                    capture_output=True, text=True, timeout=180)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
