"""Official Mamba4Rec block math with the dependency imported lazily.

MambaLayer and FeedForward are exact source slices from the pinned MIT source.
The Mamba factory changes only dependency loading, not the native Mamba1 block.
"""
import torch
from torch import nn


def Mamba(*args, **kwargs):
    from mamba_ssm.modules.mamba_simple import Mamba as NativeMamba

    return NativeMamba(*args, **kwargs)


class MambaLayer(nn.Module):
    def __init__(self, d_model, d_state, d_conv, expand, dropout, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.mamba = Mamba(
                # This module uses roughly 3 * expand * d_model^2 parameters
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
        self.dropout = nn.Dropout(dropout)
        self.LayerNorm = nn.LayerNorm(d_model, eps=1e-12)
        self.ffn = FeedForward(d_model=d_model, inner_size=d_model*4, dropout=dropout)
    
    def forward(self, input_tensor):
        hidden_states = self.mamba(input_tensor)
        if self.num_layers == 1:        # one Mamba layer without residual connection
            hidden_states = self.LayerNorm(self.dropout(hidden_states))
        else:                           # stacked Mamba layers with residual connections
            hidden_states = self.LayerNorm(self.dropout(hidden_states) + input_tensor)
        hidden_states = self.ffn(hidden_states)
        return hidden_states

class FeedForward(nn.Module):
    def __init__(self, d_model, inner_size, dropout=0.2):
        super().__init__()
        self.w_1 = nn.Linear(d_model, inner_size)
        self.w_2 = nn.Linear(inner_size, d_model)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.LayerNorm = nn.LayerNorm(d_model, eps=1e-12)

    def forward(self, input_tensor):
        hidden_states = self.w_1(input_tensor)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)

        hidden_states = self.w_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states
