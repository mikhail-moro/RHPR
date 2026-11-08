import unittest
import torch
from torch import nn
from rhpr.gr.projected_codebooks import projected_lookup


def quantizer():
    q=nn.Module();q.quantization_layer_list=nn.ModuleList([nn.Module()for _ in range(3)])
    for m in q.quantization_layer_list:m.centroids=nn.Parameter(torch.randn(256,64))
    return q


class ProjectedCodebooksTests(unittest.TestCase):
    def test_projection_preserves_initial_vectors_suffix_and_rng(self):
        q=quantizer();old=torch.randn(1024,128);rng=torch.get_rng_state().clone()
        table=projected_lookup(q,old)
        torch.testing.assert_close(torch.get_rng_state(),rng,rtol=0,atol=0)
        books=torch.cat([l.centroids for l in q.quantization_layer_list])
        torch.testing.assert_close(table.weight[:768],torch.nn.functional.pad(books,(0,64)),rtol=0,atol=0)
        torch.testing.assert_close(table.weight[768:],old[768:],rtol=0,atol=0)
        self.assertIs(table.quantizer.quantization_layer_list[0].centroids,q.quantization_layer_list[0].centroids)

    def test_frozen_centroids_receive_no_gradient_but_projection_and_suffix_learn(self):
        q=quantizer();q.requires_grad_(False);table=projected_lookup(q,torch.randn(1024,128))
        table(torch.tensor([1,257,513,769])).square().sum().backward()
        self.assertTrue(table.projection.weight.grad.abs().sum()>0)
        self.assertTrue(table.suffix.grad.abs().sum()>0)
        self.assertTrue(all(p.grad is None for p in q.parameters()))
        # The projection changes input vectors, not the assignment centroids.
        before=[l.centroids.detach().clone()for l in q.quantization_layer_list]
        torch.optim.SGD(table.parameters(),lr=.01).step()
        for a,l in zip(before,q.quantization_layer_list):torch.testing.assert_close(a,l.centroids,rtol=0,atol=0)


class AdaptationBridgeTests(unittest.TestCase):
    def test_native_endpoint_widening_preserves_function_without_new_projection(self):
        from rhpr.gr.codebook_adaptation import widen_quantizer,direct_lookup
        q=quantizer();q.encoder=nn.Module();q.encoder.model=nn.Sequential(nn.Linear(16,20),nn.ReLU(),nn.Linear(20,64));q.encoder.output_dim=64
        q.decoder=nn.Module();q.decoder.model=nn.Sequential(nn.Linear(64,30),nn.ReLU(),nn.Linear(30,16));q.decoder.input_dim=64
        x=torch.randn(7,16);z=q.encoder.model(x);old_reconstruction=q.decoder.model(z)
        original=[l.centroids.detach().clone()for l in q.quantization_layer_list]
        widen_quantizer(q);wide=q.encoder.model(x)
        torch.testing.assert_close(wide[:,:64],z,rtol=0,atol=0)
        self.assertEqual(float(wide[:,64:].abs().sum()),0.)
        torch.testing.assert_close(q.decoder.model(wide),old_reconstruction,rtol=1e-6,atol=1e-6)
        for before,l in zip(original,q.quantization_layer_list):torch.testing.assert_close(l.centroids,torch.nn.functional.pad(before,(0,64)),rtol=0,atol=0)
        lookup=direct_lookup(q,torch.randn(1024,128));lookup(torch.tensor([1,257,513])).sum().backward()
        self.assertTrue(all(l.centroids.grad is not None for l in q.quantization_layer_list))
        self.assertFalse(hasattr(lookup,'projection'))

    def test_adaptation_contract_rejects_extra_arms_or_loss_tuning(self):
        import json,copy
        from pathlib import Path
        from rhpr.gr.codebook_adaptation import validate
        c=json.loads((Path(__file__).resolve().parents[1]/'experiments/ml1m_codebook_adaptation_2t4.json').read_text());validate(c)
        for field in ['arms','loss']:
            bad=copy.deepcopy(c)
            if field=='arms':bad['trials'].append(dict(bad['trials'][0]))
            else:bad['auxiliary']['reconstruction_weight']=.1
            with self.assertRaises(ValueError):validate(bad)
