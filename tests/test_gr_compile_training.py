import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import torch
import test_gr_profiling as fixture_module
from rhpr.gr.experiment import _train_generator

class CompileTrainingTest(unittest.TestCase):
    def test_compiled_forward_updates_original_weights_and_keeps_unwrapped_checkpoints(self):
        f=fixture_module.ProfileExecutionTests();f.setUpClass()
        try:
            f.setUp();c=dict(seed=42,data=dict(history_items=2),training={**f.training,'physical_batch_size':4,'precision':'fp32','evaluation_precision':'fp32','max_epochs':2,'min_epochs':2,'patience':9,'validation_interval':1,'evaluation_batch_size':2,'max_seconds':60,'compile_training':{'backend':'eager','fullgraph':False}})
            before={k:v.clone()for k,v in f.model.state_dict().items()}
            def fake_eval(*args,**kwargs):return {'ndcg@10':.1,'seconds':0},f.np.zeros((8,2,4),dtype=int),f.np.zeros((8,2))
            with tempfile.TemporaryDirectory()as tmp,patch('rhpr.gr.experiment._evaluate',fake_eval):
                _train_generator(f.model,None,f.index,c,Path(tmp),'cpu',example_builder=lambda *args:f.examples)
                state=torch.load(Path(tmp)/'generator-last.pt',weights_only=False)
                self.assertEqual(set(before),set(state['model']));self.assertTrue(any(not torch.equal(before[k],state['model'][k])for k in before));f.model.load_state_dict(state['model'],strict=True)
                self.assertFalse(any(k.startswith('_orig_mod.')for k in state['model']))
        finally:f.tearDownClass()
