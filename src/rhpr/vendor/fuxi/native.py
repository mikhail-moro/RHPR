# Copyright 2025-2026 USTC-StarTeam
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import Dict, Optional
import logging
import torch
from torch import nn
from torch.nn import functional as F
from rhpr.vendor.hstu import packing
from .attn import Retention, LinearPositionalChannel, LinearTemporalChannel
from .mffn import MultistageFeedforwardNeuralNetwork

class FuXiLinearBlockJagged(torch.nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        linear_hidden_dim: int,
        attention_dim: int,
        dropout_ratio: float,
        attn_dropout_ratio: float,
        num_heads: int,
        linear_activation: str,
        channel_t_config: Optional[Dict],
        channel_p_config: Optional[Dict],
        id_layer: int,
        use_rope: bool = False,
        normalization: str = "rel_bias",
        epsilon: float = 1e-6,
        ffn_multiply: float = 1,
        chunk_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        self._embedding_dim: int = embedding_dim
        self._linear_dim: int = linear_hidden_dim
        self._attention_dim: int = attention_dim
        self._dropout_ratio: float = dropout_ratio
        self._attn_dropout_ratio: float = attn_dropout_ratio
        self._num_heads: int = num_heads
        self._normalization: str = normalization
        
        query_dim = attention_dim * num_heads
        key_dim = query_dim
        value_dim = linear_hidden_dim * num_heads
        
        self._attn_dim = value_dim * (1 + (channel_p_config != None) + (channel_t_config != None))
        
        self._uvqk: torch.nn.Parameter = torch.nn.Parameter(
            torch.empty(
                (
                    embedding_dim,
                    query_dim + key_dim + value_dim + self._attn_dim,
                )
            ).normal_(mean=0, std=0.02),
        )
        
        self._linear_activation: str = linear_activation
        
        self._retention = Retention(
            head_dim = attention_dim,
            num_heads = num_heads,
            use_rope = use_rope,
            chunk_size = chunk_size,
        )
        
        self._channel_t_config = channel_t_config
        # type == 'linear'
        self._channel_t = LinearTemporalChannel(
            linear_dim = value_dim,
            num_heads = channel_t_config.get('num_heads', num_heads),
            base = channel_t_config.get('base', 2),
            start_index = channel_t_config.get('start_index', 0),
            base_stride = channel_t_config.get('base_stride', 1),
            chunk_size = chunk_size,
            use_proj = channel_t_config.get('use_proj', True),
            learnable_gamma = channel_t_config.get('learnable_gamma', False) and id_layer == 0,
            no_temporal_qk = channel_t_config.get('no_temporal_qk', False),
            use_augment_connection = channel_t_config.get('aug_current', False),
        )
        
        self._channel_p_config = channel_p_config
        # type == 'linear'
        self._channel_p = LinearPositionalChannel(
            max_seq_len = channel_p_config.get('max_sequence_length', None),
            embedding_dim = channel_p_config.get('dim', 32),
            aug_current = channel_p_config.get('aug_current', True),
            use_proj = True,
            value_dim = value_dim,
            chunk_size = chunk_size,
        )
        
        logging.info(f'Temporal channel config: {channel_t_config}')
        logging.info(f'Positional channel config: {channel_p_config}')
        
        self._mffn = MultistageFeedforwardNeuralNetwork(
            ams_output_size = self._attn_dim,
            input_size = embedding_dim,
            hidden_size = int(embedding_dim * ffn_multiply),
            output_size = embedding_dim,
            dropout_ratio = dropout_ratio,
            single_stage = False,
            epsilon = epsilon
        )
        self._mffn.init()
        
        self._eps: float = epsilon

    def _norm_input(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, normalized_shape=[self._embedding_dim], eps=self._eps)

    def _norm_attn_output(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(
            x, normalized_shape=[self._attn_dim], eps=self._eps
        )

    def forward(  # pyre-ignore [3]
        self,
        x: torch.Tensor,
        x_offsets: torch.Tensor,
        all_timestamps: Optional[torch.Tensor],
        invalid_attn_mask: torch.Tensor,
        past_lengths: torch.Tensor,
        target_timestamps: torch.Tensor,
        current_length: int,
        cache = None,
        early_lengths: Optional[torch.Tensor] = None,
        early_timestamps: Optional[torch.Tensor] = None,
        return_cache_states: bool = False,
        channel_t_buffer = None,
    ):
        B: int = past_lengths.size(0)
        n: int = current_length
        
        normed_x = self._norm_input(x)
        
        if cache is not None :
            cache_s, cache_t, cache_p = cache
        else :
            cache_s, cache_t, cache_p = None, None, None
        
        batched_mm_output = torch.mm(normed_x, self._uvqk)
        batched_mm_output = F.silu(batched_mm_output)
        
        u, q, k, v = torch.split(
            batched_mm_output,
            [
                self._attn_dim,
                self._attention_dim * self._num_heads,
                self._attention_dim * self._num_heads,
                self._linear_dim * self._num_heads,
            ],
            dim=1,
        )

        B: int = x_offsets.size(0) - 1
        
        padded_v = packing.jagged_to_padded_dense(v, [x_offsets], [n])
        
        outputs = []
       
        output_latent, returned_cache = self._retention(
            q=q,
            k=k,
            v=padded_v,
            x_offsets=x_offsets,
            all_timestamps=all_timestamps,
            invalid_attn_mask=invalid_attn_mask,
            is_v_padded=True,
            current_length=current_length,
            past_lengths=past_lengths,
            cache=cache_s,
            early_lengths=early_lengths,
            early_timestamps=early_timestamps,
            return_cache_states=return_cache_states,
        )

        outputs.append(self._norm_input(output_latent))
        
        if self._channel_t != None :
            log_decay_t = torch.ones(normed_x.shape[0], self._channel_t_config.get('num_heads') * 2, dtype=torch.float32, device=normed_x.device)
            
            output_latent_t, channel_t_buffer, cache_t = self._channel_t(
                q=None,
                k=None,
                v=normed_x,
                x_offsets=x_offsets,
                all_timestamps=all_timestamps,
                invalid_attn_mask=invalid_attn_mask,
                past_lengths=past_lengths,
                is_v_padded=False,
                cache=cache_t,
                buffered_attn_map=channel_t_buffer,
                target_timestamps=target_timestamps,
                return_cache_states=return_cache_states,
                log_decay=log_decay_t,
            )
            outputs.append(self._norm_input(output_latent_t))

        if self._channel_p != None :
            output_latent_p, channel_p_buffer, cache_p = self._channel_p(
                q=q,
                k=k,
                v=normed_x,
                # w=w,
                x_offsets=x_offsets,
                all_timestamps=all_timestamps,
                invalid_attn_mask=invalid_attn_mask,
                is_v_padded=False,
                cache=cache_p,
                past_lengths=past_lengths,
                buffered_attn_map=None,
                return_cache_states=return_cache_states,
            )
            outputs.append(self._norm_input(output_latent_p))
        
        combined_output = torch.concat(
            outputs,
            dim=-1
        ).reshape(B, n, self._attn_dim)

        attn_output = packing.dense_to_jagged(
            combined_output, [x_offsets],
        )[0]
        
        attn_output = u * attn_output

        new_outputs = self._mffn(attn_output, x, past_lengths)
        
        return new_outputs, (returned_cache, cache_t, cache_p), channel_t_buffer


