"""Independent Mamba-inspired additions to the selected query SSM.

Each feature changes only one part of the compressor. The depthwise convolution
acts on normalized old-memory embeddings before both existing input projections;
it is inspired by Mamba, rather than a claim to reproduce its inner-channel
convolution exactly. The query generator and common summary readout stay intact.

Only the final valid scan output is consumed. Applying the optional D skip or
SiLU output gate there is algebraically equivalent to applying it at every scan
position and gathering the final output, without materializing K-expanded gates.
Mixed-precision rounding can differ from a fused native D/z epilogue.
"""
from __future__ import annotations

import copy
import importlib
import importlib.machinery
import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from rhpr.sasrec.query_qssm_models import build_qssm_model
from rhpr.sasrec.query_ssm import QuerySSMCompressor, _positive_integer


_FEATURES = ("conv4", "output_gate", "skip_d")
_SOURCE_VARIANT = {"name": "query_ssm_m32", "kind": "query_ssm", "summary_tokens": 32}


class MambaFeatureQuerySSMCompressor(QuerySSMCompressor):
    """One feature on top of the original zero-state query-conditioned scan.

    ``conv4`` uses a causal depthwise k4 convolution and SiLU before both u and
    input-dependent B/Delta projections. ``output_gate`` computes z from the
    original final normalized-memory event, independently of query slots, and
    multiplies the final scan read by SiLU(z). ``skip_d`` adds D*u at that event.
    The latter two share their added branch across queries; the SSM writes,
    forgetting and C readout remain query-conditioned.
    """

    def __init__(self, d_model: int, summary_tokens: int, tail_size: int = 159,
                 query_context_size: int = 128, n_heads: int = 4,
                 d_state: int = 16, d_inner: int = 128, gate_rank: int = 32,
                 write_conditioning: str = "recent", dropout: float = 0.0,
                 dt_min: float = 1e-4, dt_max: float = 0.1,
                 query_source: str = "old_suffix", *, feature: str) -> None:
        if feature not in _FEATURES:
            raise ValueError("exactly one feature is required: conv4, output_gate or skip_d")
        super().__init__(d_model, summary_tokens, tail_size, query_context_size,
                         n_heads, d_state, d_inner, gate_rank, write_conditioning,
                         dropout, dt_min, dt_max, query_source)
        self.feature = feature
        # Append only after every parent module has initialized. Never blanket
        # reinitialize this compressor: dt bias has a deliberate slow timescale.
        if feature == "conv4":
            self.history_conv = nn.Conv1d(d_model, d_model, kernel_size=4,
                                           groups=d_model, padding=0, bias=True)
        elif feature == "output_gate":
            self.output_gate = nn.Linear(d_model, d_inner)
        else:
            self.D = nn.Parameter(torch.ones(d_inner, dtype=torch.float32))
            # Metadata mirrors Mamba. The registered optimizer still decays
            # every parameter, exactly as it does in the unmodified control.
            self.D._no_weight_decay = True
        self.implementation += f"_mamba_feature_{feature}"

    def history_features(self, memory: torch.Tensor) -> torch.Tensor:
        """Transform only the occupied-prefix memory path, never query inputs."""
        if memory.ndim != 3 or memory.shape[-1] != self.d_model or memory.shape[1] < 1:
            raise ValueError("memory must be nonempty [batch,length,d_model]")
        if self.feature != "conv4":
            return memory
        channels = memory.transpose(1, 2)
        # Only left padding: output t can see input t-3,...,t. The caller
        # supplies right-masked memory and gathers each row's final valid t.
        return F.silu(self.history_conv(F.pad(channels, (3, 0)))).transpose(1, 2)

    def build_scan_inputs(self, memory: torch.Tensor,
                          queries: torch.Tensor) -> dict[str, torch.Tensor]:
        return super().build_scan_inputs(self.history_features(memory), queries)

    def endpoint_features(self, last_output: torch.Tensor,
                          last_memory: torch.Tensor,
                          last_values: torch.Tensor) -> torch.Tensor:
        """Apply the optional read epilogue before the common output projection.

        Shapes are [B,K,d_inner], [B,d_model] and [B,K,d_inner]. The skip uses
        values gathered from the already projected scan input, not a second
        projection with a separate mixed-precision execution.
        """
        if (last_output.ndim != 3 or last_output.shape[1:] !=
                (self.summary_tokens, self.d_inner)
                or last_memory.shape != (last_output.shape[0], self.d_model)
                or last_values.shape != last_output.shape):
            raise ValueError("endpoint tensors must match batch, query and channel dimensions")
        if self.feature == "skip_d":
            return (last_output.float() + self.D.float()[None, None, :]
                    * last_values.float()).to(dtype=last_output.dtype)
        if self.feature == "output_gate":
            z = self.output_gate(last_memory)
            return (last_output.float() * F.silu(z.float())[:, None, :]).to(dtype=last_output.dtype)
        return last_output

    def forward(self, embeddings: torch.Tensor, valid_mask: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        lengths, old, count, queries, query_ends, memory, _ = self._inputs(embeddings, valid_mask)
        inputs = self.build_scan_inputs(memory, queries)
        outputs = self.scan_output(inputs)
        batch, length = embeddings.shape[0], memory.shape[1]
        rows = torch.arange(batch, device=embeddings.device)[:, None]
        slots = torch.arange(self.summary_tokens, device=embeddings.device)[None, :]
        endpoints = (old - 1).clamp(min=0)
        outputs = outputs.reshape(batch, self.summary_tokens, self.d_inner,
                                  length).permute(0, 1, 3, 2)
        values = inputs["u"].reshape(batch, self.summary_tokens, self.d_inner,
                                      length).permute(0, 1, 3, 2)
        last_output = outputs[rows, slots, endpoints[:, None]]
        last_values = values[rows, slots, endpoints[:, None]]
        last_memory = memory[torch.arange(batch, device=embeddings.device), endpoints]
        last_output = self.endpoint_features(last_output, last_memory, last_values)
        readout = self.output_projection(last_output)
        return self._finish(readout, queries, embeddings, lengths, old, count, query_ends)

    def feature_compute_flops(self, old_length: int,
                              batch_size: int = 1) -> dict[str, int | str]:
        """Extra arithmetic beyond parent scan and observed Linear/MHA GEMMs.

        The existing benchmark hooks count Linear output-gate projections but
        not Conv1d or endpoint elementwise operations. Keep these separate so
        callers can add unhooked_feature_flops exactly once. MAC counts as two;
        SiLU is reported separately and bias additions are excluded.
        """
        _positive_integer(old_length, "old_length")
        _positive_integer(batch_size, "batch_size")
        conv, projection, endpoint, silu = 0, 0, 0, 0
        if self.feature == "conv4":
            conv = 2 * batch_size * old_length * self.d_model * 4
            silu = batch_size * old_length * self.d_model
        elif self.feature == "output_gate":
            projection = 2 * batch_size * self.d_model * self.d_inner
            endpoint = batch_size * self.summary_tokens * self.d_inner
            silu = batch_size * self.d_inner
        else:
            endpoint = 2 * batch_size * self.summary_tokens * self.d_inner
        return {"unhooked_feature_flops": conv + endpoint,
                "feature_conv_flops": conv,
                "hooked_feature_projection_flops": projection,
                "feature_endpoint_arithmetic_flops": endpoint,
                "feature_silu_evaluations": silu,
                "feature_cost_policy": "causal_depthwise_mac2_or_final_valid_read_epilogue"}


def build_mamba_model(config: dict[str, Any], variant: dict[str, Any],
                      trial: dict[str, Any], num_items: int) -> nn.Module:
    """Preserve the selected architecture's original fresh initializer and RNG.

    This constructs weights from scratch; the saved selected checkpoint is only
    a historical quality control. Every original tensor is copied into feature
    compressors exactly, and newly appended parameters are the only additions.
    """
    from rhpr.sasrec.query_mamba_protocol import mamba_source_config, mamba_trials
    if variant != _SOURCE_VARIANT or trial not in mamba_trials(config, variant):
        raise ValueError("a registered K32 Mamba-feature trial is required")
    source_config = mamba_source_config(config)
    source_trial = config["control_history_config"]["selected_source"]["trial"]
    model = build_qssm_model(source_config, dict(_SOURCE_VARIANT), source_trial, num_items)
    feature = trial["feature"]
    if feature == "none":
        return model
    if feature not in _FEATURES:
        raise ValueError("unknown registered Mamba feature")
    original = model.compressor
    with torch.random.fork_rng(devices=[]):
        compressor = MambaFeatureQuerySSMCompressor(d_model=original.d_model,
            summary_tokens=original.summary_tokens, tail_size=original.tail_size,
            query_context_size=original.query_context_size,
            n_heads=original.query_generator.attention.num_heads,
            d_state=original.d_state, d_inner=original.d_inner,
            gate_rank=original.gate_rank, write_conditioning=original.write_conditioning,
            dropout=original.dropout.p, dt_min=original.dt_min,
            dt_max=original.dt_max, query_source=original.query_source, feature=feature)
        added = set(compressor.state_dict()) - set(original.state_dict())
        expected = ({"history_conv.weight", "history_conv.bias"} if feature == "conv4" else
                    {"output_gate.weight", "output_gate.bias"} if feature == "output_gate" else {"D"})
        missing, unexpected = compressor.load_state_dict(original.state_dict(), strict=False)
        if added != expected or set(missing) != expected or unexpected:
            raise RuntimeError("Mamba feature changed the original compressor state contract")
    model.compressor = compressor
    return model


def audit_mamba_feature(feature: str, device: torch.device) -> dict[str, Any]:
    """Require native FP32/backward parity and finite FP16 for this feature.

    Tiny checks are independent of all recommendation data and preserve CPU and
    CUDA RNG state. A failed check aborts; no GPU Python-scan fallback exists.
    Both ragged histories contain events strictly before the old-suffix query,
    so a nonzero early-history gradient verifies the recurrence, not just Q.
    """
    from rhpr.sasrec.query_profile import _comparison, _critical_gradient_norms

    if feature not in _FEATURES:
        raise ValueError("native feature audit requires conv4, output_gate or skip_d")
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Mamba feature audit requires an actual CUDA device")
    try:
        extension = importlib.import_module("selective_scan_cuda")
        interface = importlib.import_module("mamba_ssm.ops.selective_scan_interface")
    except (ImportError, OSError, RuntimeError) as error:
        raise RuntimeError("Mamba feature audit requires the compiled native selective scan") from error
    native_path = str(getattr(extension, "__file__", ""))
    if (not any(native_path.endswith(suffix) for suffix in importlib.machinery.EXTENSION_SUFFIXES)
            or not callable(getattr(extension, "fwd", None))
            or not callable(getattr(extension, "bwd", None))
            or not callable(getattr(interface, "selective_scan_fn", None))):
        raise RuntimeError("Mamba feature audit requires compiled selective_scan_cuda fwd/bwd")

    def feature_gradients(model: nn.Module) -> dict[str, float]:
        selected = ({"history_conv.weight", "history_conv.bias"} if feature == "conv4" else
                    {"output_gate.weight", "output_gate.bias"} if feature == "output_gate" else {"D"})
        norms = {}
        for name, parameter in model.named_parameters():
            if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                raise RuntimeError(f"Mamba feature audit: missing/nonfinite gradient {name}")
            if name in selected:
                norm = float(parameter.grad.detach().float().norm())
                if not math.isfinite(norm) or norm <= 0:
                    raise RuntimeError(f"Mamba feature audit: zero/nonfinite new-feature gradient {name}")
                norms[name] = norm
        if set(norms) != selected:
            raise RuntimeError("Mamba feature audit did not find every new parameter")
        return norms

    cuda_index = device.index if device.index is not None else torch.cuda.current_device()
    with torch.random.fork_rng(devices=[cuda_index]):
        torch.random.default_generator.manual_seed(918273)
        reference = MambaFeatureQuerySSMCompressor(d_model=8, summary_tokens=2,
            tail_size=6, query_context_size=4, n_heads=2, d_state=4,
            d_inner=8, gate_rank=4, dropout=0., dt_max=.01,
            query_source="old_suffix", feature=feature).eval()
        native = copy.deepcopy(reference).to(device)
        values = torch.randn(2, 13, 8)
        mask = torch.arange(13)[None, :] < torch.tensor([13, 11])[:, None]
        values = values.masked_fill(~mask[:, :, None], 0)
        cpu_input, gpu_input = values.clone().requires_grad_(), values.to(device).requires_grad_()
        weights = torch.randn(2, 2, 8)
        expected, observed = reference(cpu_input, mask), native(gpu_input, mask.to(device))
        if reference.last_scan_backend != "cpu_reference" or native.last_scan_backend != "mamba_native_cuda":
            raise RuntimeError("Mamba feature audit executed an unexpected scan backend")
        if not torch.equal(expected[1], observed[1].cpu()) or not torch.equal(expected[2], observed[2].cpu()):
            raise RuntimeError("Mamba feature audit: packed masks/source positions differ")
        checks = [_comparison(expected[0], observed[0], name="packed_output", atol=1e-4, rtol=2e-3)]
        (expected[0][:, :2].float() * weights).sum().backward()
        (observed[0][:, :2].float() * weights.to(device)).sum().backward()
        checks.append(_comparison(cpu_input.grad, gpu_input.grad,
            name="input_gradient", atol=2e-4, rtol=3e-3))
        native_parameters = dict(native.named_parameters())
        for name, parameter in reference.named_parameters():
            if parameter.grad is None or native_parameters[name].grad is None:
                raise RuntimeError(f"Mamba feature audit: disconnected parameter {name}")
            checks.append(_comparison(parameter.grad, native_parameters[name].grad,
                name=f"parameter_gradient:{name}", atol=2e-4, rtol=3e-3))
        critical = _critical_gradient_norms(native, gpu_input, (7, 5), 4, "old_suffix")
        extra = feature_gradients(native)

        half = copy.deepcopy(reference).to(device)
        half.zero_grad(set_to_none=True)
        half_input = values.to(device).requires_grad_()
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            half_output = half(half_input, mask.to(device))[0]
        (half_output[:, :2].float() * weights.to(device)).sum().backward()
        if half.last_scan_backend != "mamba_native_cuda":
            raise RuntimeError("Mamba feature audit: FP16 used an unexpected backend")
        half_check = _comparison(observed[0], half_output,
            name="autocast_fp16_output", atol=2e-2, rtol=3e-2)
        half_critical = _critical_gradient_norms(half, half_input, (7, 5), 4, "old_suffix")
        half_extra = feature_gradients(half)
        torch.cuda.synchronize(device)
        return {"passed": True, "feature": feature, "device": str(device),
            "native_backend": native.last_scan_backend,
            "reference_backend": reference.last_scan_backend,
            "native_extension_path": native_path,
            "native_extension_fwd_bwd_verified": True,
            "python_gpu_fallback": False, "query_source": "old_suffix",
            "query_support_intervals": [[3, 7], [1, 5]],
            "entire_raw_tail_gradient_zero": True,
            "old_input_before_query_gradient_nonzero": True,
            "summary_uses_differentiable_scan_output": True,
            "checks": checks, "critical_native_gradient_norms": critical,
            "new_feature_gradient_norms": extra,
            "autocast_fp16": {"output_check": half_check,
                "all_parameter_gradients_finite": True,
                "critical_gradient_norms": half_critical,
                "new_feature_gradient_norms": half_extra},
            "tiny_shape": {"batch_size": 2, "d_model": 8,
                "summary_tokens": 2, "old_lengths": [7, 5], "raw_tail_size": 6,
                "query_context_size": 4, "d_inner": 8, "d_state": 4, "gate_rank": 4},
            "rng_preserved": True,
            "epilogue_equivalence": "algebraic; fused mixed-precision rounding need not be bit-identical"}
