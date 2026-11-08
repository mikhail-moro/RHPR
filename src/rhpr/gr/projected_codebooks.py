"""Native 64D RQ code vectors projected into TIGER's 128D input space."""
from __future__ import annotations


def projected_lookup(quantizer, original_weight):
    """Use the actual native centroid Parameters, without changing RQ geometry.

    Per-level output classifier heads remain GRID's original independent heads.
    The suffix is not a quantizer code and keeps its ordinary trainable lookup.
    """
    import torch
    from torch import nn
    class ProjectedLookup(nn.Module):
        def __init__(self):
            super().__init__()
            self.quantizer=quantizer
            self.projection=nn.Linear(64,128,bias=False,device=original_weight.device)
            self.suffix=nn.Parameter(original_weight[768:].detach().clone())
            self.num_embeddings=1024;self.embedding_dim=128
            with torch.no_grad():
                self.projection.weight.zero_()
                self.projection.weight[:64].copy_(torch.eye(64,device=original_weight.device))
        @property
        def weight(self):
            books=torch.cat([layer.centroids for layer in self.quantizer.quantization_layer_list],dim=0)
            return torch.cat([self.projection(books),self.suffix],dim=0)
        def forward(self,indices):
            return torch.nn.functional.embedding(indices,self.weight)
    if original_weight.shape!=(1024,128):raise ValueError('requires standard TIGER3x256 plus suffix,128D')
    if any(layer.centroids.shape!=(256,64)for layer in quantizer.quantization_layer_list):raise ValueError('requires native3x256x64books')
    # Constructing a deterministic bridge must not perturb dropout RNG.
    devices=[original_weight.device.index or 0]if original_weight.is_cuda else []
    with torch.random.fork_rng(devices=devices):
        return ProjectedLookup()
