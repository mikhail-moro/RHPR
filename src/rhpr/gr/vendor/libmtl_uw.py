"""Uncertainty weighting adapted from LibMTL/weighting/UW.py (MIT).

Copyright (c) 2021 median-research-group. See libmtl/LICENSE and NOTICE.md.
The upstream backward is exposed as a differentiable forward for GRID's
existing optimization loop. Missing tasks omit BOTH loss and log-variance.
"""
import torch
from torch import nn


class UncertaintyWeighting(nn.Module):
    def __init__(self, task_num):
        super().__init__()
        self.loss_scale = nn.Parameter(torch.full((task_num,), -0.5))

    def forward(self, losses, active):
        terms = losses / (2 * self.loss_scale.exp()) + self.loss_scale / 2
        return torch.where(active, terms, torch.zeros_like(terms)).sum()

    def weights(self):
        return 1 / (2 * self.loss_scale.exp())
