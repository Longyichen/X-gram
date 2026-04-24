from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from olmo_core.exceptions import OLMoConfigurationError

from ..attention import ShortConvParams, compute_injection_delta
from .ops.hash_injection import HashTokenMapInjection
from .ops.shortconv import SwiGLUShortConv
from .runtime import (
    InjectionBlockContext,
    InjectionBlockResult,
    resolve_configured_layers,
    validate_injection_layers,
)


def build_xgram_modules(
    transformer: Any,
    embedding_injection: Any,
    *,
    vocab_size: int,
    d_model: int,
    n_layers: int,
    dtype: torch.dtype,
    init_device: str,
    init_std: float,
    hash_num_heads: int,
    hash_multipliers: List[int],
) -> None:
    del init_std

    def resolve_attention_injection_dim(attention_cfg: Any) -> int:
        n_heads_cfg = getattr(attention_cfg, "n_heads", None) or 0
        n_kv_heads_cfg = getattr(attention_cfg, "n_kv_heads", None)
        n_heads = n_heads_cfg
        if n_heads <= 0:
            raise OLMoConfigurationError("Attention injection requires a valid attention.n_heads")
        n_kv_heads = n_kv_heads_cfg or n_heads
        if n_kv_heads <= 0:
            raise OLMoConfigurationError("Attention injection requires a valid attention.n_kv_heads")
        if n_heads % n_kv_heads != 0:
            raise OLMoConfigurationError(
                f"NUM_ATTN_HEADS ({n_heads}) must be divisible by n_kv_heads ({n_kv_heads}) for attention injection"
            )
        if d_model % n_heads != 0:
            raise OLMoConfigurationError(
                f"d_model ({d_model}) must be divisible by NUM_ATTN_HEADS ({n_heads}) for attention injection"
            )
        head_dim = d_model // n_heads
        return n_kv_heads * head_dim

    attention_qkv_targets = [t for t in transformer._attention_injection_targets if t in {"q", "k", "v"}]
    attention_qkv_target_set = set(attention_qkv_targets)
    attention_has_qkv = any(t in {"q", "k", "v"} for t in transformer._attention_injection_targets)
    attention_qk_sharing_active = (
        transformer._attention_qk_sharing and {"q", "k"}.issubset(attention_qkv_target_set)
    )

    default_h_layers = resolve_configured_layers(
        getattr(embedding_injection, "h_layers", None),
        default_layers=list(embedding_injection.layers),
    )
    h_injection_layers = list(default_h_layers)
    default_qkv_layers = default_h_layers
    configured_qk_layers = resolve_configured_layers(
        getattr(embedding_injection, "qk_layers", None),
        default_layers=None,
    )
    target_injection_layers: Dict[str, List[int]] = {
        "q": resolve_configured_layers(
            configured_qk_layers if attention_qk_sharing_active else getattr(embedding_injection, "q_layers", None),
            default_layers=default_qkv_layers if "q" in transformer._attention_injection_targets else [],
        ),
        "k": resolve_configured_layers(
            configured_qk_layers if attention_qk_sharing_active else getattr(embedding_injection, "k_layers", None),
            default_layers=default_qkv_layers if "k" in transformer._attention_injection_targets else [],
        ),
        "v": resolve_configured_layers(
            getattr(embedding_injection, "v_layers", None),
            default_layers=default_qkv_layers if "v" in transformer._attention_injection_targets else [],
        ),
        "o": resolve_configured_layers(
            getattr(embedding_injection, "o_layers", None),
            default_layers=default_h_layers if "o" in transformer._attention_injection_targets else [],
        ),
    }

    validate_injection_layers("INJECTION_H_LAYERS", h_injection_layers, n_layers=n_layers)
    validate_injection_layers("INJECTION_QK_LAYERS", configured_qk_layers, n_layers=n_layers)
    for target_label, target_layers in target_injection_layers.items():
        validate_injection_layers(
            f"INJECTION_{target_label.upper()}_LAYERS",
            target_layers,
            n_layers=n_layers,
        )

    if (
        transformer._attention_qk_sharing
        and configured_qk_layers
        and (getattr(embedding_injection, "q_layers", None) or getattr(embedding_injection, "k_layers", None))
    ):
        raise OLMoConfigurationError(
            "INJECTION_QK_SHARING=1 does not allow INJECTION_Q_LAYERS or INJECTION_K_LAYERS. "
            "Use INJECTION_QK_LAYERS instead."
        )
    if transformer._attention_qk_sharing and target_injection_layers["q"] != target_injection_layers["k"]:
        raise OLMoConfigurationError(
            "INJECTION_QK_SHARING=1 requires matching q/k layer occurrences. Use INJECTION_QK_LAYERS."
        )

    def qkv_container_triplet(target_label: str) -> Tuple[nn.ModuleDict, nn.ModuleDict, nn.ModuleDict]:
        if target_label == "qk":
            return (
                transformer._injection_qk_embeddings,
                transformer._injection_qk_gates,
                transformer._injection_qk_shortconvs,
            )
        if target_label == "q":
            return (
                transformer._injection_q_embeddings,
                transformer._injection_q_gates,
                transformer._injection_q_shortconvs,
            )
        if target_label == "k":
            return (
                transformer._injection_k_embeddings,
                transformer._injection_k_gates,
                transformer._injection_k_shortconvs,
            )
        if target_label == "v":
            return (
                transformer._injection_v_embeddings,
                    transformer._injection_v_gates,
                transformer._injection_v_shortconvs,
            )
        raise OLMoConfigurationError(f"Unsupported attention target label '{target_label}'")

    if attention_has_qkv:
        qkv_specs: List[Tuple[str, List[int]]] = []
        if attention_qk_sharing_active:
            qkv_specs.append(("qk", target_injection_layers["q"]))
        else:
            for target_label in attention_qkv_targets:
                qkv_specs.append((target_label, target_injection_layers[target_label]))
        if "v" in attention_qkv_target_set and all(spec[0] != "v" for spec in qkv_specs):
            qkv_specs.append(("v", target_injection_layers["v"]))

        for target_label, target_layers in qkv_specs:
            emb_dict, gate_dict, shortconv_dict = qkv_container_triplet(target_label)
            for layer_idx in target_layers:
                block_key = str(layer_idx)
                if block_key not in emb_dict:
                    emb_dict[block_key] = nn.ModuleList()
                    gate_dict[block_key] = nn.ParameterList()
                    if transformer._shortconv_enabled:
                        shortconv_dict[block_key] = nn.ModuleList()

                embedding_dim = resolve_attention_injection_dim(transformer.blocks[block_key].attention)
                module_idx = len(emb_dict[block_key])
                if transformer._hash_injection_enabled:
                    token_map_path = transformer._hash_token_map_path
                    if token_map_path is None:
                        raise OLMoConfigurationError(
                            "Attention hash injection requires hash_token_map_path in the embedding injection config"
                        )
                    hash_multipliers_vn = [
                        hash_multipliers[(module_idx * hash_num_heads + i) % len(hash_multipliers)]
                        for i in range(hash_num_heads)
                    ]
                    injection_embedding = HashTokenMapInjection(
                        vocab_size=vocab_size,
                        d_model=embedding_dim,
                        dtype=dtype,
                        init_device=init_device,
                        token_map_path=Path(token_map_path),
                        num_buckets=None,
                        top_k_count=None,
                        hash_multipliers=hash_multipliers_vn,
                    )
                else:
                    injection_embedding = nn.Embedding(
                        vocab_size,
                        embedding_dim,
                        dtype=dtype,
                        device=init_device,
                    )
                emb_dict[block_key].append(injection_embedding)

                gate_param = nn.Parameter(torch.empty(1, dtype=torch.float32, device=init_device))
                if not gate_param.is_meta:
                    gate_param.data.fill_(transformer._injection_lambda_init)
                gate_dict[block_key].append(gate_param)
                gate_idx = len(gate_dict[block_key]) - 1
                transformer._register_injection_depth_scale(
                    block_key=f"{target_label}_{block_key}",
                    gate_idx=gate_idx,
                    layer_idx=layer_idx,
                    device=init_device,
                    dtype=torch.float32,
                )

                if transformer._shortconv_enabled:
                    conv_idx = len(shortconv_dict[block_key])
                    kernel_size = transformer._sc_multi_scale_kernels[
                        conv_idx % len(transformer._sc_multi_scale_kernels)
                    ]
                    shared_conv = SwiGLUShortConv(
                        embedding_dim,
                        kernel_size=kernel_size,
                        device=init_device,
                        dtype=dtype,
                    )
                    if not shared_conv.conv_content.weight.is_meta:
                        shared_conv.conv_content.weight.data.zero_()
                        shared_conv.conv_content.weight.data[:, :, -1] = 1.0
                        nn.init.normal_(shared_conv.conv_gate.weight, mean=0.0, std=transformer.init_std)
                        if shared_conv.conv_gate.bias is not None:
                            shared_conv.conv_gate.bias.data.fill_(transformer._sc_swiglu_gate_bias)
                    shortconv_dict[block_key].append(shared_conv)

    if "o" in transformer._attention_injection_targets:
        for layer_idx in target_injection_layers["o"]:
            block_key = str(layer_idx)
            if block_key not in transformer._injection_o_embeddings:
                transformer._injection_o_embeddings[block_key] = nn.ModuleList()
                transformer._injection_o_gates[block_key] = nn.ParameterList()
                transformer._injection_o_shortconvs[block_key] = nn.ModuleList()
            if transformer._hash_injection_enabled:
                o_module_idx = len(transformer._injection_o_embeddings[block_key])
                token_map_path = transformer._hash_token_map_path
                if token_map_path is None:
                    raise OLMoConfigurationError(
                        "Output hash injection requires hash_token_map_path in the embedding injection config"
                    )
                o_hash_multipliers = [
                    hash_multipliers[(o_module_idx * hash_num_heads + i) % len(hash_multipliers)]
                    for i in range(hash_num_heads)
                ]
                o_emb = HashTokenMapInjection(
                    vocab_size=vocab_size,
                    d_model=d_model,
                    dtype=dtype,
                    init_device=init_device,
                    token_map_path=Path(token_map_path),
                    num_buckets=None,
                    top_k_count=None,
                    hash_multipliers=o_hash_multipliers,
                )
            else:
                o_emb = nn.Embedding(vocab_size, d_model, dtype=dtype, device=init_device)
            transformer._injection_o_embeddings[block_key].append(o_emb)
            o_gate = nn.Parameter(torch.empty(1, dtype=torch.float32, device=init_device))
            if not o_gate.is_meta:
                o_gate.data.fill_(transformer._injection_lambda_init)
            o_gate_idx = len(transformer._injection_o_gates[block_key])
            transformer._injection_o_gates[block_key].append(o_gate)
            transformer._register_injection_depth_scale(
                block_key=f"o_{block_key}",
                gate_idx=o_gate_idx,
                layer_idx=layer_idx,
                device=init_device,
                dtype=torch.float32,
            )
            if transformer._shortconv_enabled:
                conv_idx = len(transformer._injection_o_shortconvs[block_key])
                kernel_size = transformer._sc_multi_scale_kernels[
                    conv_idx % len(transformer._sc_multi_scale_kernels)
                ]
                conv_o = SwiGLUShortConv(
                    d_model,
                    kernel_size=kernel_size,
                    device=init_device,
                    dtype=dtype,
                )
                if not conv_o.conv_content.weight.is_meta:
                    conv_o.conv_content.weight.data.zero_()
                    conv_o.conv_content.weight.data[:, :, -1] = 1.0
                    nn.init.normal_(conv_o.conv_gate.weight, mean=0.0, std=transformer.init_std)
                    if conv_o.conv_gate.bias is not None:
                        conv_o.conv_gate.bias.data.fill_(transformer._sc_swiglu_gate_bias)
                transformer._injection_o_shortconvs[block_key].append(conv_o)

    if transformer._h_target_enabled:
        for layer_idx in h_injection_layers:
            block_key = str(layer_idx)
            if block_key not in transformer._injection_h_embeddings:
                transformer._injection_h_embeddings[block_key] = nn.ModuleList()
                if transformer._shortconv_enabled:
                    transformer._injection_h_gates[block_key] = nn.ParameterList()
                    transformer._injection_h_shortconvs[block_key] = nn.ModuleList()
                else:
                    transformer._injection_h_gates[block_key] = nn.ParameterList()

            embedding_dim = d_model
            if transformer._hash_injection_enabled:
                token_map_path = transformer._hash_token_map_path
                module_idx = len(transformer._injection_h_embeddings[block_key])
                if token_map_path is None:
                    raise OLMoConfigurationError(
                        "Hidden-state hash injection requires hash_token_map_path in the embedding injection config"
                    )
                hash_multipliers_vn = [
                    hash_multipliers[(module_idx * hash_num_heads + i) % len(hash_multipliers)]
                    for i in range(hash_num_heads)
                ]
                injection_embedding = HashTokenMapInjection(
                    vocab_size=vocab_size,
                    d_model=embedding_dim,
                    dtype=dtype,
                    init_device=init_device,
                    token_map_path=Path(token_map_path),
                    num_buckets=None,
                    top_k_count=None,
                    hash_multipliers=hash_multipliers_vn,
                )
            else:
                injection_embedding = nn.Embedding(
                    vocab_size,
                    embedding_dim,
                    dtype=dtype,
                    device=init_device,
                )

            gate_value = transformer._injection_lambda_init
            gate_param = nn.Parameter(torch.empty(1, dtype=torch.float32, device=init_device))
            if not gate_param.is_meta:
                gate_param.data.fill_(gate_value)
            transformer._injection_h_gate_defaults.setdefault(block_key, []).append(gate_value)
            transformer._injection_h_gates[block_key].append(gate_param)
            gate_idx = len(transformer._injection_h_gates[block_key]) - 1
            transformer._register_injection_depth_scale(
                block_key=block_key,
                gate_idx=gate_idx,
                layer_idx=layer_idx,
                device=init_device,
                dtype=torch.float32,
            )

            if transformer._shortconv_enabled:
                conv_idx = len(transformer._injection_h_shortconvs[block_key])
                kernel_size = transformer._sc_multi_scale_kernels[
                    conv_idx % len(transformer._sc_multi_scale_kernels)
                ]
                conv = SwiGLUShortConv(
                    d_model,
                    kernel_size=kernel_size,
                    device=init_device,
                    dtype=dtype,
                )
                if not conv.conv_content.weight.is_meta:
                    conv.conv_content.weight.data.zero_()
                    conv.conv_content.weight.data[:, :, -1] = 1.0
                    nn.init.normal_(conv.conv_gate.weight, mean=0.0, std=transformer.init_std)
                    if conv.conv_gate.bias is not None:
                        conv.conv_gate.bias.data.fill_(transformer._sc_swiglu_gate_bias)
                transformer._injection_h_shortconvs[block_key].append(conv)

            transformer._injection_h_embeddings[block_key].append(injection_embedding)


def _collect_depth_scales(
    transformer: Any,
    *,
    block_key_prefix: str,
    gate_count: int,
    block_idx: int,
    device: torch.device,
    dtype: torch.dtype,
) -> List[torch.Tensor]:
    depth_scales: List[torch.Tensor] = []
    for gate_idx in range(gate_count):
        depth_scale = transformer._get_injection_depth_scale(
            block_key=block_key_prefix,
            gate_idx=gate_idx,
            device=device,
            dtype=dtype,
        )
        if depth_scale is None:
            depth_scale = transformer._register_injection_depth_scale(
                block_key=block_key_prefix,
                gate_idx=gate_idx,
                layer_idx=block_idx,
                device=device,
                dtype=dtype,
            )
        depth_scales.append(depth_scale.to(dtype=dtype, device=device))
    return depth_scales


def prepare_xgram_block_kwargs(
    transformer: Any,
    context: InjectionBlockContext,
    *,
    warmup_scale_tensor: torch.Tensor,
) -> InjectionBlockResult:
    h = context.hidden_states
    block_kwargs: Dict[str, Any] = {}
    block_key = context.block_key
    block_idx = context.block_idx
    input_ids = context.input_ids

    if block_key in transformer._injection_h_embeddings and transformer._h_target_enabled:
        h_shortconvs = None
        if block_key in transformer._injection_h_shortconvs and len(transformer._injection_h_shortconvs[block_key]) > 0:
            h_shortconvs = transformer._injection_h_shortconvs[block_key]
        h_gates = transformer._injection_h_gates[block_key] if block_key in transformer._injection_h_gates else []
        h_depth_scales = _collect_depth_scales(
            transformer,
            block_key_prefix=block_key,
            gate_count=len(h_gates),
            block_idx=block_idx,
            device=h.device,
            dtype=h.dtype,
        )
        total_injection, injection_count, last_gate_h, last_lambda_raw_h = compute_injection_delta(
            embeddings=transformer._injection_h_embeddings[block_key],
            gates=h_gates,
            shortconvs=h_shortconvs,
            depth_scales=h_depth_scales,
            input_ids=input_ids,
            warmup_scale=warmup_scale_tensor,
            target_device=h.device,
            target_dtype=h.dtype,
            shortconv_params=ShortConvParams(rmsnorm_eps=transformer._shortconv_rmsnorm_eps),
            disable_depth_scale=transformer._injection_depth_scale_disabled,
        )
        if injection_count > 0 and total_injection is not None:
            h_prev_log = h
            h = h + total_injection.to(dtype=h.dtype, device=h.device)
            transformer._log_injection_metrics(
                h_prev=h_prev_log,
                injection_delta=total_injection,
                gate=last_gate_h,
                lambda_raw=last_lambda_raw_h,
                input_embedding=context.input_embedding,
                layer_idx=block_idx,
                step=context.step,
                warmup_scale=warmup_scale_tensor,
            )

    attention_has_qkv = any(
        t in {"q", "k", "v"} for t in getattr(transformer, "_attention_injection_targets", [])
    )
    if attention_has_qkv:
        shortconv_params = ShortConvParams(rmsnorm_eps=transformer._shortconv_rmsnorm_eps)
        block_kwargs["input_ids"] = input_ids
        if block_key in transformer._injection_qk_embeddings and transformer._injection_qk_embeddings[block_key]:
            qk_shortconvs = None
            if block_key in transformer._injection_qk_shortconvs and len(transformer._injection_qk_shortconvs[block_key]) > 0:
                qk_shortconvs = transformer._injection_qk_shortconvs[block_key]
            qk_depth_scales = _collect_depth_scales(
                transformer,
                block_key_prefix=f"qk_{block_key}",
                gate_count=len(transformer._injection_qk_gates[block_key]),
                block_idx=block_idx,
                device=h.device,
                dtype=h.dtype,
            )
            delta_qk, _, _, _ = compute_injection_delta(
                embeddings=transformer._injection_qk_embeddings[block_key],
                gates=transformer._injection_qk_gates[block_key],
                shortconvs=qk_shortconvs,
                depth_scales=qk_depth_scales,
                input_ids=input_ids,
                warmup_scale=warmup_scale_tensor,
                target_device=h.device,
                target_dtype=h.dtype,
                shortconv_params=shortconv_params,
                disable_depth_scale=transformer._injection_depth_scale_disabled,
            )
            block_kwargs["_injection_qk_delta"] = delta_qk

        target_module_specs = []
        if transformer._injection_q_embeddings is not None:
            target_module_specs.append(("q", transformer._injection_q_embeddings, transformer._injection_q_gates, transformer._injection_q_shortconvs))
        if transformer._injection_k_embeddings is not None:
            target_module_specs.append(("k", transformer._injection_k_embeddings, transformer._injection_k_gates, transformer._injection_k_shortconvs))
        target_module_specs.append(("v", transformer._injection_v_embeddings, transformer._injection_v_gates, transformer._injection_v_shortconvs))

        for target_label, emb_dict, gate_dict, shortconv_dict in target_module_specs:
            if block_key not in emb_dict or not emb_dict[block_key]:
                continue
            depth_scales = _collect_depth_scales(
                transformer,
                block_key_prefix=f"{target_label}_{block_key}",
                gate_count=len(gate_dict[block_key]),
                block_idx=block_idx,
                device=h.device,
                dtype=h.dtype,
            )
            target_shortconvs = None
            if block_key in shortconv_dict and len(shortconv_dict[block_key]) > 0:
                target_shortconvs = shortconv_dict[block_key]
            delta_target, _, _, _ = compute_injection_delta(
                embeddings=emb_dict[block_key],
                gates=gate_dict[block_key],
                shortconvs=target_shortconvs,
                depth_scales=depth_scales,
                input_ids=input_ids,
                warmup_scale=warmup_scale_tensor,
                target_device=h.device,
                target_dtype=h.dtype,
                shortconv_params=shortconv_params,
                disable_depth_scale=transformer._injection_depth_scale_disabled,
            )
            block_kwargs[f"_injection_{target_label}_delta"] = delta_target

        block_kwargs["_injection_warmup_scale"] = warmup_scale_tensor
        block_kwargs.setdefault("_injection_sc_rmsnorm_eps", transformer._shortconv_rmsnorm_eps)
        block_kwargs.setdefault("_injection_targets", transformer._injection_targets)

    if block_key in transformer._injection_o_embeddings and transformer._injection_o_embeddings[block_key]:
        o_gates = transformer._injection_o_gates[block_key] if block_key in transformer._injection_o_gates else []
        o_shortconvs = None
        if block_key in transformer._injection_o_shortconvs and len(transformer._injection_o_shortconvs[block_key]) > 0:
            o_shortconvs = transformer._injection_o_shortconvs[block_key]
        o_depth_scales = _collect_depth_scales(
            transformer,
            block_key_prefix=f"o_{block_key}",
            gate_count=len(o_gates),
            block_idx=block_idx,
            device=h.device,
            dtype=h.dtype,
        )
        delta_o, _, _, _ = compute_injection_delta(
            embeddings=transformer._injection_o_embeddings[block_key],
            gates=o_gates,
            shortconvs=o_shortconvs,
            depth_scales=o_depth_scales,
            input_ids=input_ids,
            warmup_scale=warmup_scale_tensor,
            target_device=h.device,
            target_dtype=h.dtype,
            shortconv_params=ShortConvParams(rmsnorm_eps=transformer._shortconv_rmsnorm_eps),
            disable_depth_scale=transformer._injection_depth_scale_disabled,
        )
        block_kwargs["_injection_o_delta"] = delta_o
        block_kwargs["input_ids"] = input_ids
        block_kwargs.setdefault("_injection_warmup_scale", warmup_scale_tensor)
        block_kwargs.setdefault("_injection_sc_rmsnorm_eps", transformer._shortconv_rmsnorm_eps)
        block_kwargs.setdefault("_injection_targets", transformer._injection_targets)

    return InjectionBlockResult(hidden_states=h, block_kwargs=block_kwargs)
