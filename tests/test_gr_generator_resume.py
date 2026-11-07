"""Resume must equal an uninterrupted CPU trajectory, including LR and Adam."""
import copy,json,random,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from rhpr.gr.movie_data import TemporalUser,temporal_examples
from rhpr.gr.movie_sasrec import ItemIDIndex,build_movie_sasrec
from rhpr.gr.experiment import _train_generator


class GeneratorResumeTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.users=[TemporalUser(1,(0,1,2,3),(1,1,2,2),4,3,5,4),TemporalUser(2,(6,7,8,9),(1,1,2,2),10,3,11,4)]
        self.config=dict(seed=42,data=dict(items=16,history_items=4),sasrec=dict(d_model=8,n_heads=2,n_layers=1,ffn_multiplier=2,dropout=.2),
            training=dict(learning_rate=.001,weight_decay=.0001,batch_size=4,physical_batch_size=2,precision='fp32',evaluation_precision='fp32',grad_clip_norm=1.,max_epochs=6,min_epochs=1,patience=99,patience_epochs=99,validation_interval=1,evaluation_batch_size=2,max_seconds=120,sampling='strict_timestamp_prefixes',validate_first_epoch=False,
                scheduler=dict(kind='reduce_on_plateau',factor=.5,patience_epochs=2,min_lr=1e-5,threshold=1e-4,start_epoch=3)),
            evaluation=dict(save_validation_predictions=True,save_validation_losses=True))
        self.index=ItemIDIndex(16)

    def model(self):
        random.seed(42);np.random.seed(42)
        return build_movie_sasrec(self.config,42,'cpu')

    @staticmethod
    def fake_eval(offset=0):
        count=offset
        def evaluate(model,examples,index,batch_size,device,precision):
            nonlocal count
            model.eval();count+=1
            # Exercise all saved RNG streams; the loss still uses real dropout.
            torch.rand(1);random.random();np.random.random()
            return dict(**{'ndcg@10':1. if count==1 else .5,'seconds':0.}),np.tile(np.arange(16),(len(examples[2]),1)),np.zeros((len(examples[2]),16))
        return evaluate

    def train_prefix(self,path):
        c=copy.deepcopy(self.config);c['training']['max_epochs']=2
        with patch('rhpr.gr.experiment._evaluate',self.fake_eval()):
            _train_generator(self.model(),self.users,self.index,c,path,'cpu',example_builder=temporal_examples)
        return c

    def test_roundtrip_preserves_last_optimizer_rng_scheduler_and_prior_best(self):
        with tempfile.TemporaryDirectory(dir='/tmp') as tmp:
            root=Path(tmp);full=root/'full';partial=root/'partial';resumed=root/'resumed'
            for p in [full,partial,resumed]:p.mkdir()
            with patch('rhpr.gr.experiment._evaluate',self.fake_eval()):
                _train_generator(self.model(),self.users,self.index,self.config,full,'cpu',example_builder=temporal_examples)
            self.train_prefix(partial)
            with patch('rhpr.gr.experiment._evaluate',self.fake_eval(offset=2)):
                r=_train_generator(self.model(),self.users,self.index,self.config,resumed,'cpu',example_builder=temporal_examples,resume_from=partial)
            a=torch.load(full/'generator-last.pt',weights_only=False);b=torch.load(resumed/'generator-last.pt',weights_only=False)
            self.assertEqual(a['epoch'],b['epoch']);self.assertEqual(a['updates'],b['updates']);self.assertEqual(a['scheduler'],b['scheduler'])
            for k in a['model']:self.assertTrue(torch.equal(a['model'][k],b['model'][k]),k)
            for k,v in a['optimizer']['state'].items():
                for name,value in v.items():
                    if isinstance(value,torch.Tensor):self.assertTrue(torch.equal(value,b['optimizer']['state'][k][name]))
                    else:self.assertEqual(value,b['optimizer']['state'][k][name])
            self.assertTrue(torch.equal(a['torch_rng'],b['torch_rng']));self.assertEqual(a['python_rng'],b['python_rng'])
            np.testing.assert_array_equal(a['numpy_rng'][1],b['numpy_rng'][1])
            self.assertEqual(r['best_epoch'],1);self.assertEqual(r['resumed_from_epoch'],2);self.assertEqual(r['continuation_epochs'],4)
            cv=json.loads((resumed/'training-curves.json').read_text());self.assertEqual([x['epoch'] for x in cv],list(range(1,7)))
            self.assertEqual(cv[3]['next_learning_rate'],.0005);self.assertEqual(cv[4]['learning_rate'],.0005)
            self.assertTrue(all('validation_loss' in x for x in cv));self.assertTrue((resumed/'validation-epoch-001.npz').exists())
            self.assertTrue(json.loads((resumed/'resume-verification.json').read_text())['rng_restored_and_verified'])

    def test_rejects_changed_optimizer_policy_and_sid_map_before_training(self):
        with tempfile.TemporaryDirectory(dir='/tmp') as tmp:
            root=Path(tmp);source=root/'source';source.mkdir();self.train_prefix(source)
            for mode in ['learning_rate','sid']:
                out=root/mode;out.mkdir();c=copy.deepcopy(self.config);idx=self.index
                if mode=='learning_rate':c['training']['learning_rate']=.01
                else:idx=ItemIDIndex(16);idx.sids=idx.sids[::-1].copy()
                with self.assertRaises(ValueError):_train_generator(self.model(),self.users,idx,c,out,'cpu',example_builder=temporal_examples,resume_from=source)

    def test_minimum_epoch_and_early_stop_keep_prior_best(self):
        with tempfile.TemporaryDirectory(dir='/tmp') as tmp:
            root=Path(tmp);source=root/'source';source.mkdir();out=root/'resumed';out.mkdir();self.train_prefix(source)
            c=copy.deepcopy(self.config);c['training'].update(max_epochs=10,min_epochs=5,patience_epochs=2)
            with patch('rhpr.gr.experiment._evaluate',self.fake_eval(offset=2)):
                r=_train_generator(self.model(),self.users,self.index,c,out,'cpu',example_builder=temporal_examples,resume_from=source)
            self.assertEqual(r['training_epochs'],5);self.assertEqual(r['stopping_reason'],'early_stopping');self.assertEqual(r['best_epoch'],1)


if __name__=='__main__':unittest.main()
