"""LR boundary values and actual Adam/dropout continuation must be preserved."""
import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from rhpr.gr.generator_resume import build_scheduler
from rhpr.gr.experiment import _train_generator
import test_gr_generator_resume as resume_fixture
from rhpr.gr.movie_data import temporal_examples

class WarmupCosineTest(unittest.TestCase):
    def test_two_epoch_warmup_and_exact_final_update_floor(self):
        parameter=torch.nn.Parameter(torch.tensor(1.))
        opt=torch.optim.Adam([parameter],lr=.01)
        spec=dict(kind='warmup_cosine',warmup_epochs=2,total_epochs=6,min_lr=.001)
        sch=build_scheduler(opt,spec,steps_per_epoch=3);lrs=[]
        for i in range(18):
            lrs.append(opt.param_groups[0]['lr']);opt.zero_grad();parameter.square().backward();opt.step();sch.step()
        self.assertAlmostEqual(lrs[0],.01/6);self.assertAlmostEqual(lrs[5],.01)
        self.assertGreater(lrs[6],lrs[7]);self.assertAlmostEqual(lrs[-1],.001)
        self.assertTrue(all(a<b for a,b in zip(lrs[:5],lrs[1:6])))
        self.assertTrue(all(a>b for a,b in zip(lrs[5:-1],lrs[6:])))

    def test_invalid_budget_and_missing_resume_state_fail(self):
        for warm,total,minimum in [(0,6,.001),(6,6,.001),(2,6,.01),(2,6,float('nan'))]:
            opt=torch.optim.Adam([torch.nn.Parameter(torch.tensor(1.))],lr=.01)
            with self.assertRaises(ValueError):build_scheduler(opt,dict(kind='warmup_cosine',warmup_epochs=warm,total_epochs=total,min_lr=minimum),steps_per_epoch=3)

    def test_actual_training_exact_resume_preserves_parameters_adam_and_lr(self):
        fixture=resume_fixture.GeneratorResumeTest();fixture.setUp();fixture.config['training']['scheduler']=dict(kind='warmup_cosine',warmup_epochs=2,total_epochs=6,min_lr=.0001)
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);full=root/'full';partial=root/'partial';resumed=root/'resumed'
            for p in (full,partial,resumed):p.mkdir()
            with patch('rhpr.gr.experiment._evaluate',fixture.fake_eval()):
                _train_generator(fixture.model(),fixture.users,fixture.index,fixture.config,full,'cpu',example_builder=temporal_examples)
            fixture.train_prefix(partial)
            with patch('rhpr.gr.experiment._evaluate',fixture.fake_eval(offset=2)):
                _train_generator(fixture.model(),fixture.users,fixture.index,fixture.config,resumed,'cpu',example_builder=temporal_examples,resume_from=partial)
            a=torch.load(full/'generator-last.pt',weights_only=False);b=torch.load(resumed/'generator-last.pt',weights_only=False)
            self.assertEqual(a['scheduler'],b['scheduler'])
            for key in a['model']:self.assertTrue(torch.equal(a['model'][key],b['model'][key]),key)
            for k,v in a['optimizer']['state'].items():
                for name,value in v.items():
                    if isinstance(value,torch.Tensor):self.assertTrue(torch.equal(value,b['optimizer']['state'][k][name]))
                    else:self.assertEqual(value,b['optimizer']['state'][k][name])
            x=json.loads((full/'training-curves.json').read_text());y=json.loads((resumed/'training-curves.json').read_text())
            self.assertEqual([(v['learning_rate'],v['next_learning_rate'])for v in x],[(v['learning_rate'],v['next_learning_rate'])for v in y])
            self.assertAlmostEqual(x[-1]['learning_rate_last_batch'],.0001)

    def test_pilot_declares_only_two_fresh_val_only_runs(self):
        from rhpr.gr.kindle_lr_pilot import validate
        root=Path(__file__).resolve().parents[1]
        c=json.loads((root/'experiments/kindle_recent20_lr_pilot_2t4.json').read_text());validate(c)
        for mutation in ('test','epochs','trials'):
            bad=copy.deepcopy(c)
            if mutation=='test':bad['evaluation']['test_after_selection']=True
            if mutation=='epochs':bad['training']['max_epochs']=120
            if mutation=='trials':bad['trials'].pop()
            with self.assertRaises(ValueError):validate(bad)
