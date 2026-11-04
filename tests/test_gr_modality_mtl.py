import unittest
import torch
from rhpr.gr.modality_mtl import ModalityMTL

class MTLTest(unittest.TestCase):
    def test_sum_and_uw_initial_gradients_and_weight_learning(self):
        for mode in ['sum','uw']:
            x=torch.nn.Parameter(torch.tensor([1.,2.]))
            b=ModalityMTL(mode,['a','b'])
            losses={'a':x.square().sum(),'b':(x-1).square().sum()}
            total=b.objective(losses)
            b.backward_tasks(losses,total,[x],x)
            torch.testing.assert_close(x.grad,torch.tensor([2.,6.]))
            if mode=='uw':torch.testing.assert_close(b.values.grad,torch.tensor([-4.,0.]))

    def test_pcgrad_removes_opposition_and_preserves_auxiliary_gradient(self):
        x=torch.nn.Parameter(torch.tensor([1.,0.]))
        b=ModalityMTL('pcgrad',['a','b'])
        losses={'a':x[0],'b':-x[0]}
        b.backward_tasks(losses,b.objective(losses)+3*x[1],[x],x)
        torch.testing.assert_close(x.grad,torch.tensor([0.,3.]))

    def test_pcgrad_private_heads_sum_and_no_conflict_matches_control(self):
        x=torch.nn.Parameter(torch.tensor([2.]));a=torch.nn.Parameter(torch.tensor([1.]));z=torch.nn.Parameter(torch.tensor([3.]))
        b=ModalityMTL('pcgrad',['a','b'])
        losses={'a':(x*a).sum(),'b':(x*z).sum()}
        b.backward_tasks(losses,b.objective(losses),[x,a,z],x)
        for p,want in [(x,4.),(a,2.),(z,2.)]:torch.testing.assert_close(p.grad,torch.tensor([want]))

    def test_gradnorm_weight_update_is_separate_and_sum_is_constant(self):
        x=torch.nn.Parameter(torch.tensor([1.,2.]))
        b=ModalityMTL('gradnorm',['a','b'])
        losses={'a':x.square().sum(),'b':(x-1).square().sum()}
        b.backward_tasks(losses,b.objective(losses),[x],x)
        torch.testing.assert_close(x.grad,torch.tensor([2.,6.]))
        self.assertTrue(torch.isfinite(b.values.grad).all())
        torch.testing.assert_close(b.weights().sum(),torch.tensor(2.))
        b.reset_phase();self.assertEqual(float(b.reference.sum()),0.)

    def test_contract(self):
        import json
        from pathlib import Path
        from rhpr.gr.movie_suite import validate,movie_assignments
        c=json.loads((Path(__file__).resolve().parents[1]/'experiments/movielens100k_mtl_2t4.json').read_text())
        validate(c);self.assertEqual(len(movie_assignments(c,2)),2)
        c['data']['history_items']=50
        with self.assertRaises(ValueError):validate(c)

    def test_diagnostics_do_not_replace_training_gradients(self):
        x=torch.nn.Parameter(torch.tensor([2.,1.]))
        b=ModalityMTL('uw',['a','b']);b.calls=499
        losses={'a':x.square().sum(),'b':(x-1).square().sum()}
        b.backward_tasks(losses,b.objective(losses),[x],x)
        torch.testing.assert_close(x.grad,torch.tensor([6.,2.]))
        self.assertEqual(len(b.history),1)
        self.assertEqual(len(b.history[0]['shared_encoder_gradient_cosines']),2)
