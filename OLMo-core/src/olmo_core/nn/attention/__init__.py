import logging
import math
from abc import abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed import DeviceMesh
from torch.distributed.tensor import Placement, Replicate, Shard
from torch.distributed.tensor.parallel import parallelize_module

from olmo_core.config import Config, DType, StrEnum
from olmo_core.distributed.parallel.tensor_parallel import SequenceParallel
from olmo_core.doc_utils import beta_feature
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.attention.kv_cache import KVCacheManager

from ..buffer_cache import BufferCache
from ..functional import l2_normalize
from ..layer_norm import LayerNorm, LayerNormConfig
from ..rope import (
    ComplexRotaryEmbedding,
    FusedRotaryEmbedding,
    RoPEConfig,
    RotaryEmbedding,
)
from ..utils import get_tp_wrappers
from .flash_attn_api import (
    dispatch_flash_attn,
    dispatch_flash_attn_qkvpacked,
    dispatch_flash_attn_with_kvcache,
    dispatch_ring_flash_attn,
    dispatch_ring_flash_attn_qkvpacked,
)
from .ring import (
    RingAttentionLlama3LoadBalancer,
    RingAttentionLoadBalancer,
    RingAttentionLoadBalancerType,
    RingAttentionZigZagLoadBalancer,
)

__all__ = [
    "AttentionType",
    "AttentionConfig",
    "AttentionBase",
    "Attention",
    "FusedAttention",
    "NormalizedAttention",
    "RingAttentionLoadBalancerType",
    "RingAttentionLoadBalancer",
    "RingAttentionZigZagLoadBalancer",
    "RingAttentionLlama3LoadBalancer",
    "ShortConvParams",
    "compute_shortconv_delta",
    "compute_injection_delta",
]

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ShortConvParams:
    rmsnorm_eps: float


def compute_shortconv_delta(
    *,
    src: torch.Tensor,
    conv: nn.Module,
    params: ShortConvParams,
    already_normalized: bool = False,
) -> torch.Tensor:
    sc_input = src
    if not already_normalized:
        variance = sc_input.pow(2).mean(dim=-1, keepdim=True)
        sc_input = sc_input * torch.rsqrt(variance + params.rmsnorm_eps)
    if hasattr(conv, "conv_content"):
        kernel_size = conv.conv_content.weight.shape[-1]
    else:
        kernel_size = conv.weight.shape[-1]
    pad = kernel_size - 1
    inj = sc_input.transpose(1, 2)
    inj = F.pad(inj, (pad, 0))
    inj = conv(inj)
    inj = inj[:, :, : src.shape[1]].transpose(1, 2)
    return inj


def compute_injection_delta(
    *,
    embeddings,
    gates,
    shortconvs,
    depth_scales,
    input_ids: Optional[torch.Tensor],
    warmup_scale: torch.Tensor,
    target_device: torch.device,
    target_dtype: torch.dtype,
    shortconv_params: ShortConvParams,
    disable_depth_scale: bool,
) -> Tuple[Optional[torch.Tensor], int, torch.Tensor, torch.Tensor]:
    last_gate = torch.ones(1, dtype=torch.float32, device=target_device)
    last_lambda_raw = torch.ones(1, dtype=torch.float32, device=target_device)
    if embeddings is None or gates is None or input_ids is None:
        return None, 0, last_gate, last_lambda_raw

    raw_injections = []
    sc_residuals = []
    embedding_list = list(embeddings)
    gate_list = list(gates)
    shortconv_list = list(shortconvs) if shortconvs is not None else []
    depth_scale_list = list(depth_scales) if depth_scales is not None else []

    for idx, embedding_layer in enumerate(embedding_list):
        raw_E = embedding_layer(input_ids)
        raw_injections.append(raw_E)

        sc_delta = torch.zeros_like(raw_E)
        if idx < len(shortconv_list):
            sc_delta = compute_shortconv_delta(
                src=raw_E,
                conv=shortconv_list[idx],
                params=shortconv_params,
                already_normalized=False,
            )
        sc_residuals.append(sc_delta)

    if not raw_injections:
        return None, 0, last_gate, last_lambda_raw

    warmup_scale = warmup_scale.to(dtype=target_dtype, device=target_device)
    delta_sum = torch.zeros_like(raw_injections[0], dtype=target_dtype)
    injection_count = 0
    for idx, gate_param in enumerate(gate_list):
        if idx >= len(raw_injections):
            break
        final_injection_i = raw_injections[idx] + sc_residuals[idx]
        lambda_raw = gate_param.to(dtype=target_dtype, device=target_device).view(1, 1, 1)
        depth_scale = depth_scale_list[idx] if idx < len(depth_scale_list) else None
        if disable_depth_scale:
            depth_scale = None
        if depth_scale is None:
            depth_scale = torch.tensor(1.0, dtype=target_dtype, device=target_device)
        else:
            depth_scale = depth_scale.to(dtype=target_dtype, device=target_device)
        gate = lambda_raw * depth_scale.view(1, 1, 1) * warmup_scale
        last_gate = gate.detach().to(dtype=torch.float32)
        last_lambda_raw = lambda_raw.detach().to(dtype=torch.float32)
        delta_sum = delta_sum + gate * final_injection_i.to(dtype=target_dtype, device=target_device)
        injection_count += 1

    if injection_count == 0:
        return None, 0, last_gate, last_lambda_raw
    return delta_sum * (1.0 / math.sqrt(injection_count)), injection_count, last_gate, last_lambda_raw


@dataclass
class SlidingWindowAttentionConfig(Config):
    pattern: List[int]
    """
    The pattern of window sizes to use for attention, repeated to cover all layers.
    A value of -1 indicates full attention. For example, a pattern of ``[4096, 4096, 4096, -1]``
    means that for each set of 4 layers, the first 3 will use a window size of 4096,
    and the last layer will use full attention.
    """

    force_full_attention_on_first_layer: bool = True
    """
    If `True`, the first transformer layer will always use full attention, regardless of the pattern.
    """

    force_full_attention_on_last_layer: bool = True
    """
    If `True`, the last transformer layer will always use full attention, regardless of the pattern.
    """

    def _get_window_size(self, layer_idx: int, n_layers: int) -> int:
        """
        Get the window size for a given layer, returning -1 for full attention.
        """
        if self.force_full_attention_on_first_layer and layer_idx == 0:
            return -1
        if self.force_full_attention_on_last_layer and layer_idx == (n_layers - 1):
            return -1

        # Adjust the layer index if the first layer is special-cased to full attention
        # (in which case the pattern is applied starting from the second layer)
        effective_layer_idx = layer_idx
        if self.force_full_attention_on_first_layer:
            effective_layer_idx -= 1

        window_size = self.pattern[effective_layer_idx % len(self.pattern)]
        if window_size <= 0 and window_size != -1:
            raise OLMoConfigurationError(
                f"Sliding window size must be positive or -1 (got {window_size})"
            )
        return window_size

    def should_use_swa(self, layer_idx: int, n_layers: int) -> bool:
        """
        Returns `True` if the given layer uses sliding window attention.
        """
        return self._get_window_size(layer_idx, n_layers) != -1

    def get_window_size(self, layer_idx: int, n_layers: int) -> int:
        """
        Get the sliding window size for a given layer.
        """
        window_size = self._get_window_size(layer_idx, n_layers)
        if window_size == -1:
            raise ValueError(f"Layer {layer_idx} is not configured for sliding window attention.")
        return window_size


class AttentionType(StrEnum):
    """
    An enumeration of the different attention implementations.
    """

    default = "default"
    """
    ➡️ :class:`Attention`
    """
    fused = "fused"
    """
    ➡️ :class:`FusedAttention`
    """
    normalized = "normalized"
    """
    ➡️ :class:`NormalizedAttention`
    """


@dataclass
class AttentionConfig(Config):
    """
    A configuration class for easily building any of the different attention modules.

    See the individual :class:`Attention` subclasses for a description of the configuration options.
    """

    name: AttentionType = AttentionType.default
    """
    The name of the implementation.
    """
    n_heads: int = 16
    n_kv_heads: Optional[int] = None
    bias: Optional[bool] = None
    rope: Optional[RoPEConfig] = None
    clip_qkv: Optional[float] = None
    qk_norm: Optional[LayerNormConfig] = None
    dropout: Optional[float] = None
    use_flash: Optional[bool] = None
    dtype: DType = DType.float32
    sliding_window: Optional[SlidingWindowAttentionConfig] = None
    use_head_qk_norm: Optional[bool] = None

    def num_params(self, d_model: int) -> int:
        """
        The number of params that the attention implementation will have once built.

        :param d_model: The model dimensionality.
        """
        n_heads = self.n_heads
        n_kv_heads = self.n_kv_heads or n_heads
        head_dim = d_model // n_heads
        bias = self.bias if self.bias is not None else self.name != AttentionType.normalized

        params = 0

        # Block attention Q projection.
        params += d_model * d_model
        if bias:
            params += d_model

        # Block attention KV projections.
        params += 2 * d_model * n_kv_heads * head_dim
        if bias:
            params += 2 * n_kv_heads * head_dim

        # Block attention QK norm.
        if self.qk_norm is not None:
            if self.use_head_qk_norm:
                params += 2 * self.qk_norm.num_params(head_dim)
            else:
                params += 2 * self.qk_norm.num_params(d_model)

        # Block attention out.
        params += d_model * d_model
        if bias:
            params += d_model

        # Block QK scaling factors.
        if self.name == AttentionType.normalized:
            head_dim = d_model // n_heads
            params += n_heads * head_dim
            params += n_kv_heads * head_dim

        return params

    def build(
        self,
        d_model: int,
        *,
        layer_idx: int,
        n_layers: int,
        init_device: str = "cpu",
        cache: Optional[BufferCache] = None,
    ) -> "AttentionBase":
        """
        Build the corresponding attention module.

        :param d_model: The model dimensionality.
        :param init_device: The device to initialize the parameters on, e.g. "cpu", "meta".
        """
        kwargs = self.as_dict(exclude_none=True, recurse=False)
        kwargs.pop("name")

        sliding_window_config: Optional[SlidingWindowAttentionConfig] = kwargs.pop(
            "sliding_window", None
        )
        if sliding_window_config is not None and sliding_window_config.should_use_swa(
            layer_idx, n_layers
        ):
            kwargs["window_size"] = sliding_window_config.get_window_size(layer_idx, n_layers)

        kwargs.update(
            dtype=kwargs.pop("dtype").as_pt(),
            d_model=d_model,
            init_device=init_device,
            cache=cache,
        )

        try:
            if self.name == "default":
                return Attention(**kwargs)
            elif self.name == "fused":
                kwargs.pop("use_flash", None)
                if "window_size" in kwargs:
                    raise OLMoConfigurationError(
                        "'window_size' is not supported with fused attention"
                    )
                return FusedAttention(**kwargs)
            elif self.name == "normalized":
                if "window_size" in kwargs:
                    raise OLMoConfigurationError(
                        "'window_size' is not supported with normalized attention"
                    )
                return NormalizedAttention(**kwargs)
            else:
                raise NotImplementedError(self.name)
        except TypeError as e:
            raise OLMoConfigurationError(
                f"invalid options for '{self.name}' {self.__class__.__name__}, {e}"
            ) from e


class AttentionBase(nn.Module):
    """
    Base class for attention modules.
    """

    @abstractmethod
    def apply_tp(
        self,
        tp_mesh: DeviceMesh,
        input_layout: Optional[Placement] = None,
        output_layout: Optional[Placement] = None,
        use_local_output: bool = True,
        float8_enabled: bool = False,
    ):
        raise NotImplementedError

    @abstractmethod
    def apply_cp(
        self,
        cp_mesh: DeviceMesh,
        load_balancer: RingAttentionLoadBalancerType,
        head_stride: int = 1,
    ):
        raise NotImplementedError


class Attention(AttentionBase):
    """
    An implementation of multi-head self-attention with support for multi-query (MQA)
    and grouped-query (GQA) attention.

    Intra-document masking is also supported by passing in the
    ``max_doc_len`` and ``cu_doc_lens`` parameters to :meth:`forward()`. Currently this requires
    `flash-attn <https://github.com/Dao-AILab/flash-attention>`_ (``use_flash=True``).

    .. seealso::
        :class:`FusedAttention` if you have flash-attn installed and you're not using MQA or GQA.

    :param d_model: The model hidden size.
    :param n_heads: The number of attention heads.
    :param n_kv_heads: The number of key and value heads, if different.
    :param bias: Include biases with linear layers.
    :param rope: The config for RoPE, if RoPE should be used.
    :param clip_qkv: Clip QKV to this value, if set.
    :param qk_norm: Configuration a layer norm for queries and keys.
    :param dropout: Dropout probability.
    :param use_flash: Use flash attention.
        This requires `flash-attn <https://github.com/Dao-AILab/flash-attention>`_ to be installed.
    :param dtype: The default data type to use for parameters.
    :param init_device: The device to initialize weights on.
    """

    def __init__(
        self,
        *,
        d_model: int,
        n_heads: int,
        n_kv_heads: Optional[int] = None,
        bias: bool = True,
        rope: Optional[RoPEConfig] = None,
        clip_qkv: Optional[float] = None,
        qk_norm: Optional[LayerNormConfig] = None,
        dropout: float = 0.0,
        use_flash: bool = False,
        window_size: Optional[int] = None,
        dtype: torch.dtype = torch.float32,
        init_device: str = "cpu",
        cache: Optional[BufferCache] = None,
        use_head_qk_norm: bool = False,
    ):
        super().__init__()

        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads or n_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = d_model // n_heads
        self.w_q = nn.Linear(d_model, d_model, bias=bias, dtype=dtype, device=init_device)
        self.w_k = nn.Linear(
            d_model, self.n_kv_heads * self.head_dim, bias=bias, dtype=dtype, device=init_device
        )
        self.w_v = nn.Linear(
            d_model, self.n_kv_heads * self.head_dim, bias=bias, dtype=dtype, device=init_device
        )
        self.w_out = nn.Linear(d_model, d_model, bias=bias, dtype=dtype, device=init_device)
        self.clip_qkv = clip_qkv
        self.dropout_p = dropout
        self.use_head_qk_norm = use_head_qk_norm

        self.q_norm: Optional[LayerNorm] = None
        self.k_norm: Optional[LayerNorm] = None
        if qk_norm is not None:
            if use_head_qk_norm:
                self.q_norm = qk_norm.build(size=self.head_dim, init_device=init_device)
                self.k_norm = qk_norm.build(size=self.head_dim, init_device=init_device)
            else:
                self.q_norm = qk_norm.build(size=d_model, init_device=init_device)
                self.k_norm = qk_norm.build(
                    size=self.n_kv_heads * self.head_dim, init_device=init_device
                )

        # Translate window size so that we only look left, not right.
        if window_size is not None:
            if not use_flash:
                raise OLMoConfigurationError(
                    f"'window_size' is only supported with 'use_flash=True' (got {use_flash})"
                )
            if window_size <= 0:
                raise OLMoConfigurationError(f"'window_size' must be positive (got {window_size})")
            # Flash attn window is [i - window_size[0], i + window_size[1]] inclusive
            self.window_size = (window_size - 1, 0)
        else:
            self.window_size = (-1, -1)

        self.rope: Optional[Union[RotaryEmbedding, ComplexRotaryEmbedding]] = None
        if rope is not None:
            if rope.name == "fused":
                raise OLMoConfigurationError(
                    f"fused RoPE is not compatible with {self.__class__.__name__}"
                )
            rope_class = rope.build(self.head_dim, cache=cache)
            assert isinstance(rope_class, (RotaryEmbedding, ComplexRotaryEmbedding))
            self.rope = rope_class

        self.use_flash = use_flash

        self._cp_pg: Optional[dist.ProcessGroup] = None
        self._cp_enabled = False
        self._cp_load_balancer: Optional[RingAttentionLoadBalancerType] = None

        self.kv_cache_manager: Optional[KVCacheManager] = None

    @property
    def cp_enabled(self) -> bool:
        return self._cp_enabled

    def sdpa(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_doc_lens: Optional[torch.Tensor] = None,
        cu_doc_lens_q: Optional[torch.Tensor] = None,
        cu_doc_lens_k: Optional[torch.Tensor] = None,
        max_doc_len: Optional[int] = None,
        max_doc_len_q: Optional[int] = None,
        max_doc_len_k: Optional[int] = None,
        local_k_slice: Optional[slice] = None,
        scale: Optional[float] = None,
        cache_leftpad: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        att: torch.Tensor

        if self.kv_cache_manager:
            if self.cp_enabled:
                raise RuntimeError(
                    f"'{self.__class__.__name__}' does not support KV caching with context parallelism"
                )
            if not self.use_flash:
                raise RuntimeError(
                    f"'{self.__class__.__name__}' requires flash (use_flash=True) for KV caching"
                )

            self.kv_cache_manager.record_leftpad(cache_leftpad)
            att = dispatch_flash_attn_with_kvcache(
                q,
                k=k,
                v=v,
                softmax_scale=scale,
                causal=True,
                window_size=self.window_size,
                k_cache=self.kv_cache_manager.k_cache,  # updated in-place
                v_cache=self.kv_cache_manager.v_cache,  # updated in-place
                cache_leftpad=self.kv_cache_manager.cache_leftpad,
                cache_seqlens=self.kv_cache_manager.cache_seqlens.expand(
                    self.kv_cache_manager.cache_leftpad.shape[0]
                ).contiguous(),
            )
            self.kv_cache_manager.update_seqlen(q.shape[1])

        elif self.cp_enabled:
            assert self._cp_pg is not None and self._cp_load_balancer is not None
            if not self.use_flash:
                raise RuntimeError(
                    f"'{self.__class__.__name__}' requires flash (use_flash=True) for context parallelism"
                )
            att = dispatch_ring_flash_attn(
                q,
                k,
                v,
                group=self._cp_pg,
                strategy=self._cp_load_balancer,
                cu_seqlens=cu_doc_lens,
                cu_seqlens_q=cu_doc_lens_q,
                cu_seqlens_k=cu_doc_lens_k,
                max_seqlen=max_doc_len,
                max_seqlen_q=max_doc_len_q,
                max_seqlen_k=max_doc_len_k,
                heads_k_stride=self._cp_head_stride,
                local_k_slice=local_k_slice,
                dropout_p=self.dropout_p,
                causal=True,
                softmax_scale=scale,
                window_size=self.window_size,
            )
        elif self.use_flash:
            att = dispatch_flash_attn(
                q,
                k,
                v,
                cu_seqlens=cu_doc_lens,
                cu_seqlens_q=cu_doc_lens_q,
                cu_seqlens_k=cu_doc_lens_k,
                max_seqlen=max_doc_len,
                max_seqlen_q=max_doc_len_q,
                max_seqlen_k=max_doc_len_k,
                dropout_p=self.dropout_p,
                softmax_scale=scale,
                causal=True,
                window_size=self.window_size,
            )
        else:
            # Fall back to PyTorch's SDPA...
            if any(
                opt is not None
                for opt in (
                    cu_doc_lens,
                    cu_doc_lens_q,
                    cu_doc_lens_k,
                    max_doc_len,
                    max_doc_len_q,
                    max_doc_len_k,
                )
            ):
                raise RuntimeError(
                    f"{self.__class__.__name__} requires flash-attn (use_flash=True) for intra-document masking"
                )

            # NOTE: PyTorch's SDPA doesn't support GQA, so we have to do this.
            # shape: (batch_size, n_heads, seq_len, head_dim)
            k = repeat_kv(k, self.n_rep)
            v = repeat_kv(v, self.n_rep)

            # PyTorch's SDPA expects the head dimension to come before the sequence dimension.
            # shape: (batch_size, n_heads, seq_len, head_dim),
            #        (batch_size, n_kv_heads, seq_len, head_dim),
            #        (batch_size, n_kv_heads, seq_len, head_dim)
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

            # shape: (batch_size, n_heads, seq_len, head_dim)
            att = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.dropout_p, is_causal=True, scale=scale
            )

            # shape: (batch_size, seq_len, n_heads, head_dim)
            att = att.transpose(1, 2).contiguous()

        return att

    def forward(
        self,
        x: torch.Tensor,
        cu_doc_lens: Optional[torch.Tensor] = None,
        cu_doc_lens_q: Optional[torch.Tensor] = None,
        cu_doc_lens_k: Optional[torch.Tensor] = None,
        max_doc_len: Optional[int] = None,
        max_doc_len_q: Optional[int] = None,
        max_doc_len_k: Optional[int] = None,
        local_k_slice: Optional[slice] = None,
        pos_sin: Optional[torch.Tensor] = None,
        pos_cos: Optional[torch.Tensor] = None,
        freqs_cis: Optional[torch.Tensor] = None,
        cache_leftpad: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Apply attention to the input.

        :param x: The input of shape ``(batch_size, seq_len, d_model)``.
        :param cu_doc_lens: Cumulative document lengths in the input ``x``, a 1D
            :class:`torch.int32` tensor that should always have one more element than there
            are documents (the first element in the tensor should always be ``0``).
            Required together with ``max_doc_len`` when using intra-document masking.
        :param max_doc_len: The maximum document length in the input ``x``.
            Required together with ``cu_doc_lens`` when using intra-document masking.

        :returns: The output of attention with shape ``(batch_size, seq_len, d_model)``.
        """
        B, T, _ = x.shape

        _injection_qk_delta = kwargs.get("_injection_qk_delta")
        _injection_qk_count = kwargs.get("_injection_qk_count")
        _injection_qk_last_gate = kwargs.get("_injection_qk_last_gate")
        _injection_qk_last_lambda_raw = kwargs.get("_injection_qk_last_lambda_raw")
        _injection_q_delta = kwargs.get("_injection_q_delta")
        _injection_q_count = kwargs.get("_injection_q_count")
        _injection_q_last_gate = kwargs.get("_injection_q_last_gate")
        _injection_q_last_lambda_raw = kwargs.get("_injection_q_last_lambda_raw")
        _injection_k_delta = kwargs.get("_injection_k_delta")
        _injection_k_count = kwargs.get("_injection_k_count")
        _injection_k_last_gate = kwargs.get("_injection_k_last_gate")
        _injection_k_last_lambda_raw = kwargs.get("_injection_k_last_lambda_raw")
        _injection_v_delta = kwargs.get("_injection_v_delta")
        _injection_v_count = kwargs.get("_injection_v_count")
        _injection_v_last_gate = kwargs.get("_injection_v_last_gate")
        _injection_v_last_lambda_raw = kwargs.get("_injection_v_last_lambda_raw")
        _injection_warmup_scale = kwargs.get("_injection_warmup_scale")
        _injection_targets = kwargs.get("_injection_targets")
        _injection_qkv_log_fn = kwargs.get("_injection_qkv_log_fn")
        _injection_log_layer_idx = kwargs.get("_injection_log_layer_idx")
        _injection_log_step = kwargs.get("_injection_log_step")
        _injection_log_interval = kwargs.get("_injection_log_interval", 100)
        qkv_injection_present = any(
            modules is not None
            for modules in (
                _injection_qk_delta,
                _injection_q_delta,
                _injection_k_delta,
                _injection_v_delta,
            )
        )

        if qkv_injection_present:
            targets = [str(token).strip().lower() for token in (_injection_targets or ()) if str(token).strip()]
            targets_set = set(t for t in targets if t in {"q", "k", "v"})
            if not targets_set:
                targets_set = {"v"}
            if _injection_qk_delta is not None:
                targets_set.update({"q", "k"})
            inject_q = "q" in targets_set
            inject_k = "k" in targets_set
            inject_v = "v" in targets_set

            q_orig, k_orig = self.w_q(x), self.w_k(x)
            v_orig = self.w_v(x)

            warmup_scale = _injection_warmup_scale
            if warmup_scale is None:
                warmup_scale = torch.tensor(1.0, dtype=torch.float32, device=x.device)
            elif not torch.is_tensor(warmup_scale):
                warmup_scale = torch.tensor(warmup_scale, dtype=torch.float32, device=x.device)
            else:
                warmup_scale = warmup_scale.to(dtype=torch.float32, device=x.device)
            last_gate = torch.ones(1, dtype=torch.float32, device=x.device)
            last_lambda_raw = torch.ones(1, dtype=torch.float32, device=x.device)

            def _refresh_log_scalars(
                count: int,
                gate: Optional[torch.Tensor],
                lambda_raw: Optional[torch.Tensor],
            ) -> None:
                nonlocal last_gate, last_lambda_raw
                if count <= 0:
                    return
                if gate is not None:
                    last_gate = gate.detach().to(dtype=torch.float32, device=x.device)
                if lambda_raw is not None:
                    last_lambda_raw = lambda_raw.detach().to(dtype=torch.float32, device=x.device)

            delta_qk = _injection_qk_delta
            count_qk = _injection_qk_count if _injection_qk_count is not None else 0
            qk_last_gate = _injection_qk_last_gate
            qk_last_lambda_raw = _injection_qk_last_lambda_raw
            _refresh_log_scalars(count_qk, qk_last_gate, qk_last_lambda_raw)
            delta_q = _injection_q_delta
            count_q = _injection_q_count if _injection_q_count is not None else 0
            q_last_gate = _injection_q_last_gate
            q_last_lambda_raw = _injection_q_last_lambda_raw
            _refresh_log_scalars(count_q, q_last_gate, q_last_lambda_raw)
            delta_k = _injection_k_delta
            count_k = _injection_k_count if _injection_k_count is not None else 0
            k_last_gate = _injection_k_last_gate
            k_last_lambda_raw = _injection_k_last_lambda_raw
            _refresh_log_scalars(count_k, k_last_gate, k_last_lambda_raw)
            delta_v = _injection_v_delta
            count_v = _injection_v_count if _injection_v_count is not None else 0
            v_last_gate = _injection_v_last_gate
            v_last_lambda_raw = _injection_v_last_lambda_raw
            _refresh_log_scalars(count_v, v_last_gate, v_last_lambda_raw)

            q = q_orig
            k = k_orig
            v = v_orig
            if delta_v is not None and inject_v:
                v = v_orig + delta_v.to(dtype=v_orig.dtype, device=v_orig.device)
            delta_k_effective = delta_k if delta_k is not None else delta_qk
            delta_q_effective = delta_q if delta_q is not None else delta_qk
            if delta_k_effective is not None and inject_k:
                k = k_orig + delta_k_effective.to(dtype=k_orig.dtype, device=k_orig.device)
            if delta_q_effective is not None and inject_q:
                q_delta = delta_q_effective.view(B, T, self.n_kv_heads, self.head_dim)
                q_delta = repeat_kv(q_delta, self.n_rep).reshape(B, T, -1)
                q = q_orig + q_delta.to(dtype=q_orig.dtype, device=q_orig.device)

            total_count = (2 * count_qk) + count_q + count_k + count_v
            if (
                total_count > 0
                and _injection_qkv_log_fn is not None
                and _injection_log_layer_idx is not None
                and _injection_log_step is not None
            ):
                delta_terms = []
                if delta_qk is not None:
                    delta_terms.extend([delta_qk, delta_qk])
                delta_terms.extend(d for d in (delta_q, delta_k, delta_v) if d is not None)
                if delta_terms:
                    _injection_qkv_log_fn(
                        h_prev=v_orig,
                        injection_delta=sum(delta_terms),
                        gate=last_gate,
                        lambda_raw=last_lambda_raw,
                        layer_idx=_injection_log_layer_idx,
                        step=_injection_log_step,
                        warmup_scale=warmup_scale,
                        log_interval=_injection_log_interval,
                    )
        else:
            # shape: (batch_size, seq_len, n_heads * head_dim),
            #        (batch_size, seq_len, n_kv_heads * head_dim),
            #        (batch_size, seq_len, n_kv_heads * head_dim)
            q, k, v = self.w_q(x), self.w_k(x), self.w_v(x)

        if self.clip_qkv is not None:
            q.clamp_(min=-self.clip_qkv, max=self.clip_qkv)
            k.clamp_(min=-self.clip_qkv, max=self.clip_qkv)
            v.clamp_(min=-self.clip_qkv, max=self.clip_qkv)

        if not self.use_head_qk_norm:
            if self.q_norm is not None:
                q = self.q_norm(q)
            if self.k_norm is not None:
                k = self.k_norm(k)

        # NOTE: use -1 instead of `n_heads` / `n_kv_heads` to infer actual local size when
        # using tensor parallelism.
        # shape: (batch_size, seq_len, n_heads (local), head_dim)
        q = q.view(B, T, -1, self.head_dim)
        # shape: (batch_size, seq_len, n_kv_heads (local), head_dim)
        k = k.view(B, T, -1, self.head_dim)
        # shape: (batch_size, seq_len, n_kv_heads (local), head_dim)
        v = v.view(B, T, -1, self.head_dim)

        if self.use_head_qk_norm:
            if self.q_norm is not None:
                q = self.q_norm(q)
            if self.k_norm is not None:
                k = self.k_norm(k)

        if self.rope is not None:
            # In context-parallel mode we must be given pre-sharded buffers
            if self.cp_enabled and pos_sin is None and pos_cos is None and freqs_cis is None:
                raise RuntimeError(
                    "RoPE buffers must be passed through to attention after being properly "
                    "sharded by the context parallel load balancer"
                )

            start_pos = self.kv_cache_manager.current_position() if self.kv_cache_manager else None
            q, k = self.rope(
                q,
                k,
                head_first=False,
                start_pos=start_pos,
                pos_sin=pos_sin,
                pos_cos=pos_cos,
                freqs_cis=freqs_cis,
            )

        # shape: (batch_size, seq_len, n_heads, head_dim)
        att = self.sdpa(
            q,
            k,
            v,
            cu_doc_lens=cu_doc_lens,
            cu_doc_lens_q=cu_doc_lens_q,
            cu_doc_lens_k=cu_doc_lens_k,
            max_doc_len=max_doc_len,
            max_doc_len_q=max_doc_len_q,
            max_doc_len_k=max_doc_len_k,
            local_k_slice=local_k_slice,
            cache_leftpad=cache_leftpad,
        )

        # shape: (batch_size, seq_len, d_model)
        att = att.view(B, T, -1)

        # shape: (batch_size, seq_len, d_model)
        output = self.w_out(att)

        _inj_o_delta = kwargs.get("_injection_o_delta")
        if _inj_o_delta is not None:
            _inj_o_count = kwargs.get("_injection_o_count")
            _inj_o_last_gate = kwargs.get("_injection_o_last_gate")
            _inj_o_last_lambda_raw = kwargs.get("_injection_o_last_lambda_raw")
            warmup_scale = kwargs.get("_injection_warmup_scale")
            if warmup_scale is None:
                warmup_scale = torch.tensor(1.0, dtype=torch.float32, device=output.device)
            elif not torch.is_tensor(warmup_scale):
                warmup_scale = torch.tensor(warmup_scale, dtype=torch.float32, device=output.device)
            else:
                warmup_scale = warmup_scale.to(dtype=torch.float32, device=output.device)
            log_fn = kwargs.get("_injection_o_log_fn") or kwargs.get("_injection_qkv_log_fn")
            log_layer = kwargs.get("_injection_log_layer_idx")
            log_step = kwargs.get("_injection_log_step")
            log_input_emb = kwargs.get("_injection_log_input_embedding")
            inj_o_count = _inj_o_count if _inj_o_count is not None else 0
            if inj_o_count > 0:
                if (
                    log_fn is not None
                    and log_layer is not None
                    and _inj_o_last_gate is not None
                ):
                    log_fn(
                        h_prev=output,
                        injection_delta=_inj_o_delta,
                        gate=_inj_o_last_gate,
                        lambda_raw=_inj_o_last_lambda_raw,
                        input_embedding=log_input_emb,
                        layer_idx=log_layer,
                        step=log_step,
                        warmup_scale=warmup_scale,
                        log_interval=_injection_log_interval,
                    )
                output = output + _inj_o_delta.to(dtype=output.dtype, device=output.device)

        return output

    def apply_tp(
        self,
        tp_mesh: DeviceMesh,
        input_layout: Optional[Placement] = None,
        output_layout: Optional[Placement] = None,
        use_local_output: bool = True,
        float8_enabled: bool = False,
    ):
        rowwise_parallel, colwise_parallel, prepare_module_input = get_tp_wrappers(
            float8_enabled=float8_enabled
        )

        parallelize_module(
            self,
            device_mesh=tp_mesh,
            parallelize_plan=prepare_module_input(
                input_layouts=None if input_layout is None else (input_layout,),
                desired_input_layouts=(Replicate(),),
            ),
        )

        plan = {
            "w_q": colwise_parallel(
                output_layouts=None if self.q_norm is None else Shard(1),
                use_local_output=self.q_norm is None,
            ),
            "w_k": colwise_parallel(
                output_layouts=None if self.k_norm is None else Shard(1),
                use_local_output=self.k_norm is None,
            ),
            "w_v": colwise_parallel(),
            "w_out": rowwise_parallel(
                output_layouts=output_layout, use_local_output=use_local_output
            ),
        }
        if self.q_norm is not None:
            # if full-dim norm: output is sharded on the embedding dimension (B, T, E [sharded])
            #    which will be reshaped into (B, T, H [sharded], D)
            # if head-wise norm: output is sharded on the head dimension (B, T, H [sharded], D)
            plan["q_norm"] = SequenceParallel(use_local_output=True, output_layouts=Shard(2))
        if self.k_norm is not None:
            plan["k_norm"] = SequenceParallel(use_local_output=True, output_layouts=Shard(2))

        parallelize_module(
            module=self,
            device_mesh=tp_mesh,
            parallelize_plan=plan,
        )

    def apply_cp(
        self,
        cp_mesh: DeviceMesh,
        load_balancer: RingAttentionLoadBalancerType,
        head_stride: int = 1,
    ):
        """
        Prepare the module for context-parallelism (ring attention).

        .. important::
            This requires flash-attn and ring-flash-attn (``use_flash=True``).

        :param cp_mesh: The context parallel device sub-mesh.
        :param load_balancer: The load balancer type.
        """
        self._cp_pg = cp_mesh.get_group()
        self._cp_load_balancer = load_balancer
        self._cp_enabled = True
        self._cp_head_stride = head_stride

    def init_kv_cache_manager(self, batch_size: int, max_seq_len: int):
        """
        Initialize the kv cache manager for attention. When the kv cache manager exists,
        kv caching will be used during the forward pass. This should only be called during inference.

        :param batch_size: The batch size for the cache.
        :param max_seq_len: The maximum sequence length for the cache.
        """
        self.kv_cache_manager = KVCacheManager(
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            num_kv_heads=self.n_kv_heads,
            head_dim=self.head_dim,
            device=self.w_k.weight.device,
        )


@beta_feature
class NormalizedAttention(Attention):
    """
    An nGPT attention implementation.
    """

    def __init__(
        self,
        *,
        d_model: int,
        n_heads: int,
        n_kv_heads: Optional[int] = None,
        rope: Optional[RoPEConfig] = None,
        qk_norm: Optional[LayerNormConfig] = None,
        use_flash: bool = False,
        dtype: torch.dtype = torch.float32,
        init_device: str = "cpu",
        cache: Optional[BufferCache] = None,
    ):
        super().__init__(
            d_model=d_model,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            rope=rope,
            qk_norm=qk_norm,
            use_flash=use_flash,
            bias=False,
            dtype=dtype,
            init_device=init_device,
            cache=cache,
        )

        self.sq_init_value = 1.0
        self.sq_init_scaling = 1.0 / math.sqrt(d_model)
        self.sq = nn.Parameter(
            torch.empty(self.head_dim * self.n_heads, dtype=dtype, device=init_device)
        )

        self.sk_init_value = 1.0
        self.sk_init_scaling = 1.0 / math.sqrt(d_model)
        self.sk = nn.Parameter(
            torch.empty(self.head_dim * self.n_kv_heads, dtype=dtype, device=init_device)
        )

        self.sqrt_head_dim = math.sqrt(self.head_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.sq)
        nn.init.ones_(self.sk)
        with torch.no_grad():
            self.sq.mul_(self.sq_init_scaling)
            self.sk.mul_(self.sk_init_scaling)

    def forward(
        self,
        x: torch.Tensor,
        cu_doc_lens: Optional[torch.Tensor] = None,
        cu_doc_lens_q: Optional[torch.Tensor] = None,
        cu_doc_lens_k: Optional[torch.Tensor] = None,
        max_doc_len: Optional[int] = None,
        max_doc_len_q: Optional[int] = None,
        max_doc_len_k: Optional[int] = None,
        local_k_slice: Optional[slice] = None,
        pos_sin: Optional[torch.Tensor] = None,
        pos_cos: Optional[torch.Tensor] = None,
        freqs_cis: Optional[torch.Tensor] = None,
        cache_leftpad: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if cache_leftpad:
            raise NotImplementedError(
                "cache_leftpad is not supported for the normalized attention variant"
            )

        B, T, _ = x.shape

        # shape: (batch_size, seq_len, n_heads * head_dim),
        #        (batch_size, seq_len, n_kv_heads * head_dim),
        #        (batch_size, seq_len, n_kv_heads * head_dim)
        q, k, v = self.w_q(x), self.w_k(x), self.w_v(x)

        if self.q_norm is not None and self.k_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        sq = (self.sq * (self.sq_init_value / self.sq_init_scaling)).view(1, 1, -1)
        q = sq * q

        sk = (self.sk * (self.sk_init_value / self.sk_init_scaling)).view(1, 1, -1)
        k = sk * k

        # shape: (batch_size, seq_len, n_heads, head_dim)
        q = q.view(B, T, self.n_heads, self.head_dim)
        # shape: (batch_size, seq_len, n_kv_heads, head_dim)
        k = k.view(B, T, self.n_kv_heads, self.head_dim)
        # shape: (batch_size, seq_len, n_kv_heads, head_dim)
        v = v.view(B, T, self.n_kv_heads, self.head_dim)

        if self.rope is not None:
            if self.cp_enabled and pos_sin is None and pos_cos is None and freqs_cis is None:
                raise RuntimeError(
                    "RoPE buffers must be passed through to attention after being properly "
                    "sharded by the context parallel load balancer"
                )

            start_pos = self.kv_cache_manager.current_position() if self.kv_cache_manager else None
            q, k = self.rope(
                q,
                k,
                head_first=False,
                start_pos=start_pos,
                pos_sin=pos_sin,
                pos_cos=pos_cos,
                freqs_cis=freqs_cis,
            )

        # shape: (batch_size, seq_len, n_heads, head_dim)
        att = self.sdpa(
            q,
            k,
            v,
            cu_doc_lens=cu_doc_lens,
            cu_doc_lens_q=cu_doc_lens_q,
            cu_doc_lens_k=cu_doc_lens_k,
            max_doc_len=max_doc_len,
            max_doc_len_q=max_doc_len_q,
            max_doc_len_k=max_doc_len_k,
            local_k_slice=local_k_slice,
            scale=self.sqrt_head_dim,
            cache_leftpad=cache_leftpad,
        )

        # shape: (batch_size, seq_len, d_model)
        att = att.view(B, T, -1)

        # shape: (batch_size, seq_len, d_model)
        return self.w_out(att)

    def apply_tp(
        self,
        tp_mesh: DeviceMesh,
        input_layout: Optional[Placement] = None,
        output_layout: Optional[Placement] = None,
        use_local_output: bool = True,
        float8_enabled: bool = False,
    ):
        del tp_mesh, input_layout, output_layout, use_local_output, float8_enabled

        raise NotImplementedError("TP is not implemented yet for the normalized attention variant")

    @torch.no_grad()
    def normalize_matrices(self):
        """
        Normalize the weights in all matrices. This should be called after each optimizer step, which
        the :class:`~olmo_core.train.train_module.TransformerTrainModule` will handle for you.
        """
        self._normalize_matrix(self.w_q.weight)
        self._normalize_matrix(self.w_k.weight)
        self._normalize_matrix(self.w_v.weight)
        self._normalize_matrix(self.w_out.weight, dim=0)

    def _normalize_matrix(self, w: torch.Tensor, dim: int = -1):
        w.copy_(l2_normalize(w, dim=dim))


class FusedAttention(AttentionBase):
    """
    An "fused" implementation of multi-head self-attention.

    Intra-document masking is supported by passing in the ``max_doc_len`` and ``cu_doc_lens``
    parameters to :meth:`forward()`.

    .. warning::
        This requires `flash-attn <https://github.com/Dao-AILab/flash-attention>`_ to be installed.

    .. warning::
        If using RoPE, this requires that you use the "fused" RoPE implementation
        (:class:`~olmo_core.nn.rope.FusedRotaryEmbedding`).

    :param d_model: The model hidden size.
    :param n_heads: The number of attention heads.
    :param bias: Include biases with linear layers.
    :param rope: The config for RoPE, if RoPE should be used.
    :param clip_qkv: Clip QKV to this value, if set.
    :param dropout: Dropout probability.
    :param dtype: The default data type to use for parameters.
    :param init_device: The device to initialize weights on.
    """

    def __init__(
        self,
        *,
        d_model: int,
        n_heads: int,
        bias: bool = True,
        rope: Optional[RoPEConfig] = None,
        clip_qkv: Optional[float] = None,
        dropout: float = 0.0,
        dtype: torch.dtype = torch.float32,
        init_device: str = "cpu",
        cache: Optional[BufferCache] = None,
    ):
        super().__init__()

        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.w_qkv = nn.Linear(d_model, 3 * d_model, bias=bias, dtype=dtype, device=init_device)
        self.w_out = nn.Linear(d_model, d_model, bias=bias, dtype=dtype, device=init_device)
        self.clip_qkv = clip_qkv
        self.dropout_p = dropout
        self.rope: Optional[FusedRotaryEmbedding] = None
        if rope is not None:
            if rope.name != "fused":
                raise OLMoConfigurationError(f"{self.__class__.__name__} requires fused RoPE")
            rope_class = rope.build(self.head_dim, cache=cache)
            assert isinstance(rope_class, FusedRotaryEmbedding)
            self.rope = rope_class

        self._cp_pg: Optional[dist.ProcessGroup] = None
        self._cp_enabled = False
        self._cp_load_balancer: Optional[RingAttentionLoadBalancerType] = None

    @property
    def cp_enabled(self) -> bool:
        return self._cp_enabled

    def forward(
        self,
        x: torch.Tensor,
        max_doc_len: Optional[int] = None,
        cu_doc_lens: Optional[torch.Tensor] = None,
        pos_sin: Optional[torch.Tensor] = None,
        pos_cos: Optional[torch.Tensor] = None,
        freqs_cis: Optional[torch.Tensor] = None,
        cache_leftpad: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply attention to the input.

        :param x: The input of shape ``(batch_size, seq_len, d_model)``.
        :param max_doc_len: The maximum document length in the input ``x``.
            Required together with ``cu_doc_lens`` when using intra-document masking.
        :param cu_doc_lens: Cumulative document lengths in the input ``x``, a 1D
            :class:`torch.int32` tensor that should always have one more element than there
            are documents (the first element in the tensor should always be ``0``).
            Required together with ``max_doc_len`` when using intra-document masking.

        :returns: The output of attention with shape ``(batch_size, seq_len, d_model)``.
        """
        if cache_leftpad:
            raise NotImplementedError(
                "cache_leftpad is not supported for the fused attention variant"
            )

        B, T, _ = x.shape

        # shape: (batch_size, seq_len, 3, n_heads, head_dim)
        qkv = self.w_qkv(x).view(B, T, 3, self.n_heads, self.head_dim)

        if self.clip_qkv is not None:
            qkv.clamp_(min=-self.clip_qkv, max=self.clip_qkv)

        if self.rope is not None:
            if self.cp_enabled and pos_sin is None and pos_cos is None and freqs_cis is None:
                raise RuntimeError(
                    "RoPE buffers must be passed through to attention after being properly "
                    "sharded by the context parallel load balancer"
                )
            qkv = self.rope(qkv, pos_sin=pos_sin, pos_cos=pos_cos, freqs_cis=freqs_cis)

        if self.cp_enabled:
            assert self._cp_pg is not None and self._cp_load_balancer is not None
            att = dispatch_ring_flash_attn_qkvpacked(
                qkv,
                group=self._cp_pg,
                strategy=self._cp_load_balancer,
                cu_seqlens=cu_doc_lens,
                max_seqlen=max_doc_len,
                dropout_p=self.dropout_p,
                causal=True,
            )
        else:
            att = dispatch_flash_attn_qkvpacked(
                qkv,
                cu_seqlens=cu_doc_lens,
                max_seqlen=max_doc_len,
                dropout_p=self.dropout_p,
                causal=True,
            )

        # shape: (batch_size, seq_len, d_model)
        att = att.view(B, T, -1)  # type: ignore

        # shape: (batch_size, seq_len, d_model)
        return self.w_out(att)

    def apply_tp(
        self,
        tp_mesh: DeviceMesh,
        input_layout: Optional[Placement] = None,
        output_layout: Optional[Placement] = None,
        use_local_output: bool = True,
        float8_enabled: bool = False,
    ):
        del tp_mesh, input_layout, output_layout, use_local_output, float8_enabled

        raise NotImplementedError("TP is not implemented yet for the fused attention variant")

    def apply_cp(
        self,
        cp_mesh: DeviceMesh,
        load_balancer: RingAttentionLoadBalancerType,
        head_stride: int = 1,
    ):
        self._cp_pg = cp_mesh.get_group()
        self._cp_load_balancer = load_balancer
        self._cp_enabled = True
        self._cp_head_stride = head_stride


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        torch.unsqueeze(x, dim=3)
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )
