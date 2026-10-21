"""Common next-item API for the pinned official Mamba4Rec block stack.

The recommendation FFN/residual math is vendored; Mamba1 itself is supplied by
the native mamba-ssm installation. Production never substitutes a Python scan.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
from pathlib import Path

import torch
from torch import nn

from rhpr.vendor.mamba4rec.native import MambaLayer


MAMBA4REC_UPSTREAM_REVISION = "44a0c8432b34e3b89d46035b10e8525b1bd368dc"
MAMBA4REC_IMPLEMENTATION = "official_mamba4rec_with_native_mamba1_cuda"


def _require_native_cuda_runtime() -> dict[str, object]:
    """Fail before measurement if the Mamba1 CUDA extension cannot be loaded.

    Construction can happen on CPU before .to(cuda), but actual native forward
    still requires CUDA tensors. Optional causal-conv1d controls whether native
    Mamba uses its fused inner kernel or selective scan plus PyTorch convolution.
    Neither path is a reference/Python scan fallback.
    """
    try:
        scan = importlib.import_module("selective_scan_cuda")
        simple = importlib.import_module("mamba_ssm.modules.mamba_simple")
    except (ImportError, OSError, RuntimeError) as error:
        raise RuntimeError(
            "Mamba4Rec requires a compatible mamba-ssm Mamba1 CUDA installation "
            "with selective_scan_cuda; no Python/reference timing fallback is permitted"
        ) from error
    if not callable(getattr(scan, "fwd", None)) or not callable(getattr(scan, "bwd", None)):
        raise RuntimeError("selective_scan_cuda must expose native fwd and bwd functions")
    if not hasattr(simple, "Mamba"):
        raise RuntimeError("mamba-ssm must expose the Mamba1 mamba_simple.Mamba API")
    causal_conv_available = callable(getattr(simple, "causal_conv1d_fn", None))
    try:
        version = importlib.metadata.version("mamba-ssm")
    except importlib.metadata.PackageNotFoundError:
        version = "unavailable"
    try:
        causal_conv_version = importlib.metadata.version("causal-conv1d")
    except importlib.metadata.PackageNotFoundError:
        causal_conv_version = "unavailable"
    source_file = getattr(simple, "__file__", None)
    source_sha = (hashlib.sha256(Path(source_file).read_bytes()).hexdigest()
                  if source_file and Path(source_file).is_file() else "unavailable")
    return {
        "mamba_ssm_version": version,
        "causal_conv1d_version": causal_conv_version,
        "selective_scan_cuda_available": True,
        "causal_conv1d_available": causal_conv_available,
        "native_mamba1_source_sha256": source_sha,
        "execution_path": ("native_fused_mamba_inner_cuda" if causal_conv_available
                           else "native_selective_scan_cuda_with_pytorch_depthwise_conv"),
        "python_reference_fallback": False,
    }


class Mamba4RecEncoder(nn.Module):
    """Official recommendation blocks with right-padding output masking."""

    def __init__(self, *, d_model: int, n_layers: int, dropout: float,
                 d_state: int, d_conv: int, expand: int,
                 layer_norm_eps: float) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            MambaLayer(d_model=d_model, d_state=d_state, d_conv=d_conv,
                       expand=expand, dropout=dropout, num_layers=n_layers)
            for _ in range(n_layers)
        ])
        for layer in self.layers:
            layer.LayerNorm.eps = layer_norm_eps
            layer.ffn.LayerNorm.eps = layer_norm_eps

    def forward(self, encoded: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            encoded = layer(encoded)
            encoded = encoded.masked_fill(~valid_mask.unsqueeze(-1), 0)
        return encoded


class Mamba4RecModel(nn.Module):
    """Official Mamba4Rec input policy and Mamba1 + FFN blocks.

    No absolute positional embedding is added: order comes from the causal
    convolution and scan, as in the original recommendation implementation.
    Only wrapper embeddings/FFNs receive recommendation initialization, leaving
    all native SSM initialization intact, including dt_proj.bias._no_reinit.
    """

    def __init__(self, *, num_items: int, max_seq_len: int, d_model: int,
                 n_layers: int, dropout: float, n_heads: int = 1,
                 ffn_multiplier: int = 4, compressor_type: str = "identity",
                 compression_depth: int = 0, tail_size: int = 128,
                 use_time_features: bool = False, num_gap_buckets: int = 33,
                 initializer_range: float = 0.02, layer_norm_eps: float = 1e-12,
                 mamba_d_state: int = 32, mamba_d_conv: int = 4,
                 mamba_expand: int = 2) -> None:
        super().__init__()
        dimensions = (num_items, max_seq_len, d_model, n_layers, n_heads,
                      tail_size, mamba_d_state, mamba_d_conv, mamba_expand)
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 1
               for value in dimensions):
            raise ValueError("Mamba4Rec dimensions must be positive integers")
        if compressor_type != "identity":
            raise ValueError("Mamba4Rec requires identity/full-history inputs")
        if ffn_multiplier != 4:
            raise ValueError("official Mamba4Rec requires ffn_multiplier=4")
        if use_time_features:
            raise ValueError("Mamba4Rec profile requires Time OFF")
        if not 0 <= dropout < 1 or initializer_range <= 0 or layer_norm_eps <= 0:
            raise ValueError("dropout must be in [0,1), initialization and norm eps positive")
        self.native_runtime = _require_native_cuda_runtime()
        self.architecture = "mamba4rec"
        self.attention_implementation = self.sequence_implementation = MAMBA4REC_IMPLEMENTATION
        self.upstream_revision = MAMBA4REC_UPSTREAM_REVISION
        self.num_items, self.max_seq_len, self.d_model = num_items, max_seq_len, d_model
        self.n_heads, self.n_layers, self.ffn_multiplier = n_heads, n_layers, ffn_multiplier
        self.compressor_type, self.compression_depth, self.tail_size = "identity", 0, tail_size
        self.use_time_features, self.num_gap_buckets = False, num_gap_buckets
        self.gap_embedding = self.position_embedding = None
        self.mamba_d_state, self.mamba_d_conv, self.mamba_expand = mamba_d_state, mamba_d_conv, mamba_expand
        self.layer_norm_eps = layer_norm_eps
        self.item_embedding = nn.Embedding(num_items + 1, d_model, padding_idx=0)
        self.input_dropout = nn.Dropout(dropout)
        self.input_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        # The official recommendation blocks use 1e-12 in every LayerNorm.
        # Keep caller-requested eps consistent throughout the wrapper, while the
        # profile uses the official 1e-12 default.
        self.encoder = Mamba4RecEncoder(
            d_model=d_model, n_layers=n_layers, dropout=dropout,
            d_state=mamba_d_state, d_conv=mamba_d_conv, expand=mamba_expand,
            layer_norm_eps=layer_norm_eps,
        )
        self._initialize_recommendation_parameters(initializer_range)

    @property
    def mamba_layers(self) -> nn.ModuleList:
        return self.encoder.layers

    @torch.no_grad()
    def _initialize_recommendation_parameters(self, initializer_range: float) -> None:
        nn.init.normal_(self.item_embedding.weight, std=initializer_range)
        self.item_embedding.weight[0].zero_()
        for layer in self.mamba_layers:
            # Do not recurse into layer.mamba: dt projection, A_log, D and all
            # native SSM projections retain their own constructor initialization.
            for module in layer.ffn.modules():
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, std=initializer_range)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

    def encode(self, item_ids: torch.Tensor, valid_mask: torch.Tensor,
               gap_ids: torch.Tensor | None = None
               ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if gap_ids is not None:
            raise ValueError("Mamba4Rec Time OFF does not accept gap_ids")
        if item_ids.ndim != 2 or item_ids.shape != valid_mask.shape or valid_mask.dtype != torch.bool:
            raise ValueError("item_ids and boolean valid_mask must have matching [batch,length] shapes")
        lengths = valid_mask.sum(dim=1)
        if torch.any(lengths < 1):
            raise ValueError("empty sequences are not supported")
        if item_ids.shape[1] > self.max_seq_len:
            raise ValueError("raw sequence exceeds max_seq_len")
        positions = torch.arange(item_ids.shape[1], device=item_ids.device)[None, :]
        if not torch.equal(valid_mask, positions < lengths[:, None]):
            raise ValueError("Mamba4Rec requires contiguous histories with right padding")
        # Ignore arbitrary values carried in padding slots before lookup.
        embeddings = self.item_embedding(item_ids.masked_fill(~valid_mask, 0))
        embeddings = self.input_norm(self.input_dropout(embeddings))
        encoded = embeddings.masked_fill(~valid_mask.unsqueeze(-1), 0)
        # Right padding follows all real states, so it cannot affect their
        # causal computation. Mask after each layer for clean output/gradients.
        encoded = self.encoder(encoded, valid_mask)
        sources = positions.expand(item_ids.shape[0], -1).masked_fill(~valid_mask, -1)
        return encoded, valid_mask, sources

    def last_state(self, item_ids: torch.Tensor, valid_mask: torch.Tensor,
                   gap_ids: torch.Tensor | None = None) -> torch.Tensor:
        encoded, output_mask, _ = self.encode(item_ids, valid_mask, gap_ids)
        indexes = output_mask.sum(dim=1) - 1
        return encoded[torch.arange(encoded.shape[0], device=encoded.device), indexes]


MambaModel = Mamba4RecModel
