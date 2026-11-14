import copy
import importlib.util
import json
import unittest
from pathlib import Path

import numpy as np

from rhpr.gr.kindle_mixed_views import MixedViews, POLICY, locate_resume, validate


class MixedViewsTest(unittest.TestCase):
    def test_contract_keeps_one_fit_and_prior_training_protocol(self):
        root = Path(__file__).resolve().parents[1]
        c = json.loads((root/'experiments/kindle_mixed_views.json').read_text())
        validate(c)
        for key,value in [('full_probability',1.),('unit','item'),('inference','full32')]:
            changed = copy.deepcopy(c); changed['mixed_views'][key] = value
            with self.assertRaises(ValueError):
                validate(changed)
        changed = copy.deepcopy(c); changed['training']['max_epochs'] = 120
        with self.assertRaises(ValueError):
            validate(changed)

    def test_native_shapes_all_original_examples_once_without_global_rng_changes(self):
        n = 1031
        full = np.repeat(np.arange(n)[:,None],128,axis=1)
        pair = np.repeat(np.arange(n)[:,None],96,axis=1)
        views = MixedViews(full,pair,batch_size=8)
        np.random.seed(910)
        state = np.random.get_state()
        widths = set()
        order = np.random.default_rng(43).permutation(n)
        for b,begin in enumerate(range(0,n,8)):
            chosen = order[begin:begin+8]
            # Two microbatches must use the same representation.
            batch_widths = set()
            for position in range(0,len(chosen),3):
                part = chosen[position:position+3]
                actual = views(1,b,part)
                np.testing.assert_array_equal(actual[:,0],part)
                batch_widths.add(actual.shape[1]); widths.add(actual.shape[1])
            self.assertEqual(len(batch_widths),1)
        summary = views.epoch_summary(1)
        self.assertEqual(sum(summary['examples'].values()),n)
        self.assertEqual(sum(summary['batches'].values()),129)
        self.assertEqual(widths,{96,128})
        after = np.random.get_state()
        np.testing.assert_array_equal(state[1],after[1])
        self.assertEqual(state[2:],after[2:])
        np.testing.assert_array_equal(views.schedule(9),MixedViews(full,pair,batch_size=8).schedule(9))
        self.assertFalse(np.array_equal(views.schedule(1),views.schedule(2)))

    def test_audit_rejects_lost_or_duplicated_training_examples(self):
        views = MixedViews(np.zeros((4,128)),np.zeros((4,96)),batch_size=4)
        views(1,0,np.array([0,1,1,3]))
        with self.assertRaises(RuntimeError):
            views.epoch_summary(1)
        with self.assertRaises(RuntimeError):
            views(2,0,np.arange(4))

    def test_cancelled_parent_requires_pinned_artifacts_and_unchanged_training(self):
        import hashlib
        import tempfile
        root = Path(__file__).resolve().parents[1]
        c = json.loads((root/'experiments/kindle_mixed_views_resume.json').read_text())
        validate(c)
        original = json.loads((root/'experiments/kindle_mixed_views.json').read_text())
        with tempfile.TemporaryDirectory() as tmp:
            mounted = Path(tmp)/'rhpr-kindle-mixed-views'; mounted.mkdir()
            content = json.dumps(original).encode(); (mounted/'experiment.json').write_bytes(content)
            c['resume_parent']['files'] = {'experiment.json':dict(size_bytes=len(content),sha256=hashlib.sha256(content).hexdigest())}
            self.assertEqual(locate_resume(c,Path(tmp)),mounted)
            changed = copy.deepcopy(c); changed['mixed_views']['full_probability'] = .25
            with self.assertRaisesRegex(ValueError,'resume changes mixed_views'):
                locate_resume(changed,Path(tmp))
            (mounted/'experiment.json').write_bytes(content+b' ')
            with self.assertRaisesRegex(ValueError,'resume artifact differs'):
                locate_resume(c,Path(tmp))

    @unittest.skipUnless(importlib.util.find_spec('torch'),'requires torch')
    def test_training_loop_uses_mixed_inputs_but_unchanged_targets_and_pair_validation(self):
        import contextlib
        import io
        import tempfile
        from unittest.mock import patch
        import torch
        from rhpr.gr.experiment import _train_generator
        from rhpr.gr.pair_unique4 import InputIndex4
        from rhpr.gr.sid import SidIndex

        n = 16
        # Each row has a recognizable item-specific code in both formats.
        full = np.repeat(np.arange(n)[:,None],128,axis=1).astype(np.int32)
        pair = np.repeat(np.arange(n)[:,None]+1024,96,axis=1).astype(np.int32)
        users = np.arange(n)+1; targets = np.arange(n)
        sids = np.column_stack([targets,np.zeros((n,3),dtype=int)])
        wrapped = InputIndex4(SidIndex.from_full_sids(sids,width=256))
        views = MixedViews(full,pair,batch_size=2)
        seen = []
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.weight = torch.nn.Parameter(torch.ones(()))
            def rhpr_loss(self,batch):
                seen.append((batch['input_ids'].clone(),batch['future_ids'].clone(),batch['user_id'].clone()))
                return self.weight.square()
        c = dict(seed=42,data=dict(history_items=32),training=dict(batch_size=2,physical_batch_size=2,
            learning_rate=.001,weight_decay=.0001,precision='fp32',evaluation_precision='fp32',
            max_epochs=2,min_epochs=2,patience=5,max_seconds=60,grad_clip_norm=1.,validation_interval=1,
            evaluation_batch_size=2))
        evaluation_inputs = []
        def evaluate(model,examples,*args,**kwargs):
            evaluation_inputs.append(examples[0].shape[1])
            return {'ndcg@10':.1},np.zeros((n,20,4)),np.zeros((n,20))
        with tempfile.TemporaryDirectory() as tmp, patch('rhpr.gr.experiment._evaluate',side_effect=evaluate), contextlib.redirect_stdout(io.StringIO()):
            result = _train_generator(Model(),None,wrapped,c,Path(tmp),'cpu',
                example_builder=lambda *args:(pair,users,targets),training_context_selector=views)
            curves = json.loads((Path(tmp)/'training-curves.json').read_text())
        self.assertEqual(result['optimizer_updates'],16)
        self.assertEqual(evaluation_inputs,[96,96])
        self.assertEqual({x.shape[1] for x,_,_ in seen},{96,128})
        for x,y,u in seen:
            np.testing.assert_array_equal(x[:,0].numpy()%1024,y[:,0].numpy())
            np.testing.assert_array_equal(u[:,0].numpy()-1,y[:,0].numpy())
        self.assertTrue(all(r['training_views']['examples_consumed_once'] for r in curves))

    @unittest.skipUnless(importlib.util.find_spec('torch'),'requires torch')
    def test_epoch_boundary_resume_preserves_dropout_adam_lr_and_view_sequence(self):
        import contextlib
        import io
        import random
        import tempfile
        from unittest.mock import patch
        import torch
        from rhpr.gr.experiment import _train_generator
        from rhpr.gr.pair_unique4 import InputIndex4
        from rhpr.gr.sid import SidIndex

        n = 16
        full = np.repeat(np.arange(n)[:,None],128,axis=1).astype(np.int32)
        pair = np.repeat(np.arange(n)[:,None]+1024,96,axis=1).astype(np.int32)
        users = np.arange(n)+1; targets = np.arange(n)
        sids = np.column_stack([targets,np.zeros((n,3),dtype=int)])
        wrapped = InputIndex4(SidIndex.from_full_sids(sids,width=256))
        c = dict(seed=42,data=dict(history_items=32),mixed_views=POLICY,
            training=dict(batch_size=2,physical_batch_size=2,learning_rate=.001,weight_decay=.0001,
                precision='fp32',evaluation_precision='fp32',max_epochs=4,min_epochs=4,patience=99,
                max_seconds=60,grad_clip_norm=1.,validation_interval=1,evaluation_batch_size=2,
                scheduler=dict(kind='warmup_cosine',warmup_epochs=1,total_epochs=4,min_lr=.0003)))
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.embedding=torch.nn.Embedding(16,4); self.dropout=torch.nn.Dropout(.25)
            def rhpr_loss(self,batch):
                x=self.dropout(self.embedding(batch['input_ids'][:,0]%1024))
                return (x.square().mean()*(1. if batch['input_ids'].shape[1]==128 else 1.1))
        def build():
            random.seed(42); np.random.seed(42); torch.manual_seed(42)
            return Model()
        def evaluate(*args,**kwargs):
            torch.rand(1); random.random(); np.random.random()
            return {'ndcg@10':.1,'seconds':0.},np.zeros((n,20,4)),np.zeros((n,20))
        with tempfile.TemporaryDirectory() as tmp,patch('rhpr.gr.experiment._evaluate',side_effect=evaluate),contextlib.redirect_stdout(io.StringIO()):
            paths = {k:Path(tmp)/k for k in ('full','partial','resumed')}
            for p in paths.values(): p.mkdir()
            def train(config,path,resume=None):
                return _train_generator(build(),None,wrapped,config,path,'cpu',example_builder=lambda *args:(pair,users,targets),
                    training_context_selector=MixedViews(full,pair,batch_size=2),resume_from=resume)
            train(c,paths['full'])
            partial=copy.deepcopy(c); partial['training'].update(max_epochs=2,min_epochs=2)
            train(partial,paths['partial'])
            result=train(c,paths['resumed'],paths['partial'])
            a=torch.load(paths['full']/'generator-last.pt',weights_only=False)
            b=torch.load(paths['resumed']/'generator-last.pt',weights_only=False)
            self.assertEqual(result['resumed_from_epoch'],2)
            for name in a['model']: self.assertTrue(torch.equal(a['model'][name],b['model'][name]),name)
            self.assertEqual(a['scheduler'],b['scheduler'])
            self.assertEqual(a['updates'],b['updates'])
            self.assertTrue(torch.equal(a['torch_rng'],b['torch_rng']))
            for key,values in a['optimizer']['state'].items():
                for name,value in values.items():
                    self.assertTrue(torch.equal(value,b['optimizer']['state'][key][name]))
            curves=[json.loads((paths[k]/'training-curves.json').read_text()) for k in ('full','resumed')]
            self.assertEqual([r['training_views'] for r in curves[0]],[r['training_views'] for r in curves[1]])


if __name__ == '__main__':
    unittest.main()
