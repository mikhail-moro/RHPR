import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np,torch
import test_gr_generator_resume as fixture_module
from rhpr.gr.generator_resume import EMAStopping
from rhpr.gr.experiment import _train_generator
from rhpr.gr.movie_data import temporal_examples

class EMAStoppingTest(unittest.TestCase):
    def test_causal_ema_and_significant_reference_are_separate_from_raw_peak(self):
        s=EMAStopping(dict(kind='ema',alpha=.5,min_delta=.05));s.observe(.5,5);s.observe(.6,10)
        self.assertAlmostEqual(s.state['ema'],.55);self.assertEqual(s.state['last_improvement_epoch'],5)
        s.observe(.62,15);self.assertAlmostEqual(s.state['ema'],.585);self.assertEqual(s.state['last_improvement_epoch'],15)
        restored=EMAStopping(s.spec,s.state);self.assertEqual(s.observe(.4,20),restored.observe(.4,20))

    def test_resume_preserves_raw_best_model_and_exact_ema_trajectory(self):
        f=fixture_module.GeneratorResumeTest();f.setUp();f.config['training']['early_stopping']=dict(kind='ema',alpha=.5,min_delta=.0005);f.config['training']['patience_epochs']=99
        with tempfile.TemporaryDirectory()as tmp:
            r=Path(tmp);full=r/'full';partial=r/'partial';resume=r/'resume'
            for p in (full,partial,resume):p.mkdir()
            with patch('rhpr.gr.experiment._evaluate',f.fake_eval()):_train_generator(f.model(),f.users,f.index,f.config,full,'cpu',example_builder=temporal_examples)
            f.train_prefix(partial)
            with patch('rhpr.gr.experiment._evaluate',f.fake_eval(offset=2)):_train_generator(f.model(),f.users,f.index,f.config,resume,'cpu',example_builder=temporal_examples,resume_from=partial)
            a=torch.load(full/'generator-last.pt',weights_only=False);b=torch.load(resume/'generator-last.pt',weights_only=False)
            self.assertEqual(a['early_stopping_state'],b['early_stopping_state']);self.assertEqual(a['best_epoch'],1)
            for k in a['model']:self.assertTrue(torch.equal(a['model'][k],b['model'][k]))
            for folder in (full,resume):
                curve=json.loads((folder/'training-curves.json').read_text());self.assertEqual(curve[1]['validation_ema_ndcg10'],.75)

    def test_policy_rejects_invalid_smoothing_and_pilot_is_one_new_floor(self):
        for alpha,delta in [(0,0),(2,0),(.5,-1)]:
            with self.assertRaises(ValueError):EMAStopping(dict(kind='ema',alpha=alpha,min_delta=delta))
        from rhpr.gr.kindle_lr_pilot import validate
        c=json.loads((Path(__file__).resolve().parents[1]/'experiments/kindle_lr003_floor001_ema.json').read_text());validate(c)
        self.assertEqual(len(c['trials']),1);self.assertFalse(c['evaluation']['test_after_selection']);self.assertEqual(c['trials'][0]['min_lr'],.001)
