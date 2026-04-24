from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

import torch
import torch.nn as nn

from olmo_core.distributed.utils import get_local_tensor
from olmo_core.nn.moe.moe import MoEBase
from olmo_core.nn.transformer import Transformer
from olmo_core.nn.transformer.block import MoETransformerBlock


class ActivationTracker:
    """Utility to aggregate activation norms through forward hooks."""

    def __init__(self, module: nn.Module, *, pre_hook: bool):
        self._sum: Optional[torch.Tensor] = None
        self._sum_sq: Optional[torch.Tensor] = None
        self._count: Optional[torch.Tensor] = None
        self._max: Optional[torch.Tensor] = None
        self._enabled = True
        if pre_hook:
            self._handle = module.register_forward_pre_hook(self._capture_pre)
        else:
            self._handle = module.register_forward_hook(self._capture_post)

    def close(self) -> None:
        if hasattr(self, "_handle") and self._handle is not None:
            self._handle.remove()
            self._handle = None

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = enabled

    def reset(self) -> None:
        self._sum = None
        self._sum_sq = None
        self._count = None
        self._max = None

    def _capture_pre(self, module: nn.Module, inputs: Sequence[torch.Tensor]) -> None:
        del module
        if not inputs or not self._enabled:
            return
        self._update(inputs[0])

    def _capture_post(self, module: nn.Module, inputs: Sequence[torch.Tensor], output) -> None:
        del module, inputs
        if not self._enabled:
            return
        self._update(output)

    def _update(self, tensor: torch.Tensor) -> None:
        local = get_local_tensor(tensor.detach())
        if local.is_sparse:
            local = local.coalesce().values()
        flat = local.reshape(-1, local.shape[-1])
        if flat.numel() == 0:
            return
        norms = flat.float().norm(dim=-1)
        sum_norm = norms.sum()
        sum_sq_norm = norms.pow(2).sum()
        count = torch.tensor(float(norms.numel()), device=norms.device)
        max_norm = norms.max()
        if self._sum is None:
            self._sum = sum_norm
            self._sum_sq = sum_sq_norm
            self._count = count
            self._max = max_norm
        else:
            assert self._sum_sq is not None and self._count is not None and self._max is not None
            self._sum = self._sum + sum_norm
            self._sum_sq = self._sum_sq + sum_sq_norm
            self._count = self._count + count
            self._max = torch.maximum(self._max, max_norm)

    def summary(self) -> Optional[Dict[str, torch.Tensor]]:
        if self._sum is None or self._sum_sq is None or self._count is None or self._max is None:
            return None
        if self._count.item() == 0:
            return None
        mean = self._sum / self._count
        variance = torch.clamp(self._sum_sq / self._count - mean.pow(2), min=0.0)
        std = torch.sqrt(variance)
        return {
            "mean": mean.detach().cpu(),
            "std": std.detach().cpu(),
            "max": self._max.detach().cpu(),
        }


def collect_grad_norms(
    model: Transformer,
    *,
    include_layerwise: bool,
    include_modulewise: bool,
) -> Dict[str, torch.Tensor]:
    """Return L2 gradient norms for selected parameter groups."""

    grad_sq: Dict[str, torch.Tensor] = {}

    def accumulate(key: str, value: torch.Tensor) -> None:
        if key in grad_sq:
            grad_sq[key] = grad_sq[key] + value
        else:
            grad_sq[key] = value

    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        grad = get_local_tensor(param.grad.detach())
        if grad.is_sparse:
            grad = grad.coalesce().values()
        grad_float = grad.float()
        value = grad_float.pow(2).sum()
        if value.numel() == 0:
            continue

        block_prefix: Optional[str] = None
        if name.startswith("blocks."):
            parts = name.split(".")
            if len(parts) >= 2 and parts[1].isdigit():
                block_prefix = f"blocks/{int(parts[1]):02d}"

        if include_layerwise and block_prefix is not None:
            accumulate(block_prefix, value)

        if include_modulewise:
            if block_prefix is not None and ".attention.w_qkv" in name:
                qkv_chunks = _split_qkv_squares(grad_float)
                if qkv_chunks is not None:
                    for suffix, chunk_value in zip(
                        ["q_proj", "k_proj", "v_proj"], qkv_chunks
                    ):
                        accumulate(f"{block_prefix}/attention/{suffix}", chunk_value)
                    continue
            for module_key in _module_keys_from_param_name(name, block_prefix):
                accumulate(module_key, value)

    norms: Dict[str, torch.Tensor] = {}
    for key, value in grad_sq.items():
        norms[key] = torch.sqrt(value).cpu()
    return norms


def collect_param_means(
    model: Transformer,
    *,
    keywords: Sequence[str],
) -> Dict[str, torch.Tensor]:
    """Return the elementwise mean of parameters whose names match the given keywords."""

    lowered = [keyword.lower() for keyword in keywords]
    means: Dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        if not any(keyword in name.lower() for keyword in lowered):
            continue
        tensor = get_local_tensor(param.detach())
        if tensor.numel() == 0:
            continue
        total = tensor.float().sum()
        count = tensor.numel()
        if count == 0:
            continue
        mean = total / count
        raw_key = name.replace(".", "/")
        # Align naming with grad_norm: include remapped module keys (drops underscores/prefixes)
        block_prefix: Optional[str] = None
        if name.startswith("blocks."):
            parts = name.split(".")
            if len(parts) >= 2 and parts[1].isdigit():
                block_prefix = f"blocks/{int(parts[1]):02d}"
        mapped_keys = _module_keys_from_param_name(name, block_prefix)
        if not mapped_keys:
            mapped_keys = [raw_key]
        # Keep both mapped and raw to avoid losing compatibility.
        for key in set(mapped_keys + [raw_key]):
            means[key] = mean.cpu()
    return means


def _split_qkv_squares(grad: torch.Tensor) -> Optional[List[torch.Tensor]]:
    if grad.ndim == 0:
        return None
    if grad.ndim == 1:
        if grad.shape[0] % 3 != 0:
            return None
        chunks = grad.chunk(3, dim=0)
    else:
        if grad.shape[0] % 3 == 0:
            chunks = grad.chunk(3, dim=0)
        elif grad.shape[-1] % 3 == 0:
            chunks = grad.chunk(3, dim=grad.ndim - 1)
        else:
            return None
    return [chunk.pow(2).sum() for chunk in chunks]


def _module_keys_from_param_name(name: str, block_prefix: Optional[str]) -> List[str]:
    keys: List[str] = []
    parts = name.split(".")
    if not parts:
        return keys

    head = parts[0]
    if head == "embeddings":
        suffix = "weight"
        if len(parts) >= 2:
            suffix = parts[1]
            if len(parts) >= 3 and parts[2] == "bias":
                suffix = f"{suffix}/bias"
            elif len(parts) >= 3:
                suffix = f"{suffix}/{'/'.join(parts[2:])}"
        keys.append(f"embeddings/{suffix}")
        return keys

    if head == "lm_head":
        suffix = "output"
        if len(parts) >= 2:
            lm_attr = parts[1]
            mapping = {
                "w_out": "out_proj",
                "norm": "norm",
            }
            suffix = mapping.get(lm_attr, lm_attr)
            if len(parts) >= 3 and parts[2] == "bias":
                suffix = f"{suffix}/bias"
            elif len(parts) >= 3:
                suffix = f"{suffix}/{'/'.join(parts[2:])}"
        keys.append(f"lm_head/{suffix}")
        return keys

    if head == "_injection_h_embeddings":
        if len(parts) >= 2:
            layer_idx = parts[1]
            suffix = "weight"
            if len(parts) >= 3:
                suffix = parts[2]
                if len(parts) >= 4 and parts[3] == "bias":
                    suffix = f"{suffix}/bias"
                elif len(parts) >= 4:
                    suffix = f"{suffix}/{'/'.join(parts[3:])}"
            keys.append(f"injection_h_embeddings/{layer_idx}/{suffix}")
        else:
            keys.append("injection_h_embeddings")
        return keys

    if head == "_injection_h_gates":
        if len(parts) >= 2:
            layer_idx = parts[1]
            suffix = "weight"
            if len(parts) >= 3:
                suffix = parts[2]
                if len(parts) >= 4 and parts[3] == "bias":
                    suffix = f"{suffix}/bias"
                elif len(parts) >= 4:
                    suffix = f"{suffix}/{'/'.join(parts[3:])}"
            keys.append(f"injection_h_gates/{layer_idx}/{suffix}")
        else:
            keys.append("injection_h_gates")
        return keys

    for head_name, metric_prefix in (
        ("_injection_qk_embeddings", "injection_qk_embeddings"),
        ("_injection_q_embeddings", "injection_q_embeddings"),
        ("_injection_k_embeddings", "injection_k_embeddings"),
        ("_injection_v_embeddings", "injection_v_embeddings"),
        ("_injection_o_embeddings", "injection_o_embeddings"),
        ("_injection_qk_gates", "injection_qk_gates"),
        ("_injection_q_gates", "injection_q_gates"),
        ("_injection_k_gates", "injection_k_gates"),
        ("_injection_v_gates", "injection_v_gates"),
        ("_injection_o_gates", "injection_o_gates"),
    ):
        if head == head_name:
            if len(parts) >= 2:
                block_idx = parts[1]
                suffix = "weight"
                if len(parts) >= 3:
                    suffix = parts[2]
                    if len(parts) >= 4 and parts[3] == "bias":
                        suffix = f"{suffix}/bias"
                    elif len(parts) >= 4:
                        suffix = f"{suffix}/{'/'.join(parts[3:])}"
                keys.append(f"{metric_prefix}/{block_idx}/{suffix}")
            else:
                keys.append(metric_prefix)
            return keys



    for head_name, metric_prefix in (
        ("_injection_h_shortconvs", "injection_h_shortconvs"),
        ("_injection_qk_shortconvs", "injection_qk_shortconvs"),
        ("_injection_q_shortconvs", "injection_q_shortconvs"),
        ("_injection_k_shortconvs", "injection_k_shortconvs"),
        ("_injection_v_shortconvs", "injection_v_shortconvs"),
        ("_injection_o_shortconvs", "injection_o_shortconvs"),
    ):
        if head == head_name:
            if len(parts) >= 4:
                block_idx, conv_idx, conv_attr = parts[1], parts[2], parts[3]
                suffix_parts = parts[4:]
                suffix = conv_attr
                if suffix_parts:
                    suffix = f"{suffix}/{'/'.join(suffix_parts)}"
                keys.append(f"{metric_prefix}/{block_idx}/{conv_idx}/{suffix}")
            else:
                keys.append(metric_prefix)
            return keys

    if head == "_retoken_embeddings":
        if len(parts) >= 2:
            layer_idx = parts[1]
            suffix = "weight"
            if len(parts) >= 3:
                suffix = parts[2]
                if len(parts) >= 4 and parts[3] == "bias":
                    suffix = f"{suffix}/bias"
                elif len(parts) >= 4:
                    suffix = f"{suffix}/{'/'.join(parts[3:])}"
            keys.append(f"retoken_embeddings/{layer_idx}/{suffix}")
        else:
            keys.append("retoken_embeddings")
        return keys

    if head == "_retoken_scalers":
        if len(parts) >= 2:
            layer_idx = parts[1]
            suffix = "scale"
            if len(parts) >= 3:
                suffix = parts[2]
                if len(parts) >= 4:
                    suffix = f"{suffix}/{'/'.join(parts[3:])}"
            keys.append(f"retoken_scalers/{layer_idx}/{suffix}")
        else:
            keys.append("retoken_scalers")
        return keys

    if block_prefix is None or head != "blocks" or len(parts) < 3:
        return keys

    sub = parts[2]
    tail = parts[3:]

    if sub == "_injection_h_shortconvs":
        if len(tail) >= 2:
            conv_idx, conv_attr = tail[0], tail[1]
            suffix_parts = tail[2:]
            suffix = conv_attr
            if suffix_parts:
                suffix = f"{suffix}/{'/'.join(suffix_parts)}"
            keys.append(f"{block_prefix}/injection_h_shortconvs/{conv_idx}/{suffix}")
        else:
            keys.append(f"{block_prefix}/injection_h_shortconvs")
        return keys

    if sub == "attention":
        if tail:
            att_attr = tail[0]
            mapping = {
                "w_q": "q_proj",
                "w_k": "k_proj",
                "w_v": "v_proj",
                "w_out": "out_proj",
                "w_qkv": "qkv_proj",
                "q_norm": "q_norm",
                "k_norm": "k_norm",
            }
            suffix = mapping.get(att_attr, att_attr)
            extra = tail[1:]
            if extra and extra[0] == "bias":
                suffix = f"{suffix}/bias"
                extra = extra[1:]
            if extra:
                suffix = f"{suffix}/{'/'.join(extra)}"
            keys.append(f"{block_prefix}/attention/{suffix}")
        else:
            keys.append(f"{block_prefix}/attention")
        return keys

    if sub == "attention_norm":
        suffix = "attention_norm"
        if tail:
            if tail[0] == "bias":
                suffix = f"{suffix}/bias"
            else:
                suffix = f"{suffix}/{'/'.join(tail)}"
        keys.append(f"{block_prefix}/{suffix}")
        return keys

    if sub == "post_attention_norm":
        suffix = "post_attention_norm"
        if tail:
            if tail[0] == "bias":
                suffix = f"{suffix}/bias"
            else:
                suffix = f"{suffix}/{'/'.join(tail)}"
        keys.append(f"{block_prefix}/{suffix}")
        return keys

    if sub == "attn_alpha":
        keys.append(f"{block_prefix}/attention/alpha")
        return keys

    if sub == "feed_forward":
        if tail:
            ff_attr = tail[0]
            mapping = {
                "w1": "up",
                "w3": "gate",
                "w2": "down",
                "sw1": "up_scale",
                "sw3": "gate_scale",
            }
            suffix = mapping.get(ff_attr, ff_attr)
            extra = tail[1:]
            if extra and extra[0] == "bias":
                suffix = f"{suffix}/bias"
                extra = extra[1:]
            if extra:
                suffix = f"{suffix}/{'/'.join(extra)}"
            keys.append(f"{block_prefix}/ffn/{suffix}")
        else:
            keys.append(f"{block_prefix}/ffn")
        return keys

    if sub == "feed_forward_norm":
        suffix = "ffn_norm"
        if tail:
            if tail[0] == "bias":
                suffix = f"{suffix}/bias"
            else:
                suffix = f"{suffix}/{'/'.join(tail)}"
        keys.append(f"{block_prefix}/{suffix}")
        return keys

    if sub == "mlp_alpha":
        keys.append(f"{block_prefix}/ffn/alpha")
        return keys

    if sub == "feed_forward_moe":
        if tail:
            moe_attr = tail[0]
            extra = tail[1:]
            if moe_attr == "router":
                keys.append(f"{block_prefix}/moe/router")
                if extra:
                    keys.append(f"{block_prefix}/moe/router/{'/'.join(extra)}")
                return keys
            if moe_attr == "shared_mlp":
                if extra:
                    shared_attr = extra[0]
                    mapping = {
                        "w1": "up",
                        "w2": "down",
                        "w3": "gate",
                    }
                    suffix = mapping.get(shared_attr, shared_attr)
                    remainder = extra[1:]
                    if remainder and remainder[0] == "bias":
                        suffix = f"{suffix}/bias"
                        remainder = remainder[1:]
                    if remainder:
                        suffix = f"{suffix}/{'/'.join(remainder)}"
                    keys.append(f"{block_prefix}/moe/shared/{suffix}")
                else:
                    keys.append(f"{block_prefix}/moe/shared")
                return keys
            if moe_attr == "experts":
                if extra and extra[0] == "mlp":
                    remainder = extra[1:]
                    if remainder:
                        mlp_attr = remainder[0]
                        mapping = {
                            "w1": "up",
                            "w2": "down",
                            "w3": "gate",
                        }
                        suffix = mapping.get(mlp_attr, mlp_attr)
                        remainder = remainder[1:]
                        if remainder and remainder[0] == "bias":
                            suffix = f"{suffix}/bias"
                            remainder = remainder[1:]
                        if remainder:
                            suffix = f"{suffix}/{'/'.join(remainder)}"
                        keys.append(f"{block_prefix}/moe/experts/mlp/{suffix}")
                    else:
                        keys.append(f"{block_prefix}/moe/experts/mlp")
                elif extra:
                    keys.append(f"{block_prefix}/moe/experts/{'/'.join(extra)}")
                else:
                    keys.append(f"{block_prefix}/moe/experts")
                return keys
        keys.append(f"{block_prefix}/moe")
        return keys

    if sub == "feed_forward_moe_norm":
        suffix = "moe/norm"
        if tail:
            if tail[0] == "bias":
                suffix = f"{suffix}/bias"
            else:
                suffix = f"{suffix}/{'/'.join(tail)}"
        keys.append(f"{block_prefix}/{suffix}")
        return keys

    if tail:
        keys.append(f"{block_prefix}/{sub}/{'/'.join(tail)}")
    else:
        keys.append(f"{block_prefix}/{sub}")
    return keys


def filter_grad_norm_keys(norms: Dict[str, torch.Tensor]) -> Dict[str, float]:
    filtered: Dict[str, float] = {}
    for key in sorted(norms.keys()):
        normalized = key.replace("/_", "/")
        if (
            key.startswith("embeddings")
            or key.startswith("lm_head")
            or key.startswith("injection_h_embeddings")
            or key.startswith("injection_h_gates")
            or key.startswith("injection_q_embeddings")
            or key.startswith("injection_q_gates")
            or key.startswith("injection_k_embeddings")
            or key.startswith("injection_k_gates")
            or key.startswith("injection_v_embeddings")
            or key.startswith("injection_v_gates")
            or key.startswith("injection_o_embeddings")
            or key.startswith("injection_o_gates")
            or "/injection_h_shortconvs" in normalized
            or "/injection_q_shortconvs" in normalized
            or "/injection_k_shortconvs" in normalized
            or "/injection_v_shortconvs" in normalized
            or "/injection_o_shortconvs" in normalized
            or "conv" in key
            or "embedding" in key
            or key.startswith("retoken_embeddings")
            or key.startswith("retoken_scalers")
            or "/attention" in key
            or "/ffn" in key
            or "/moe" in key
        ):
            filtered[key] = float(norms[key].item())
    return filtered


def iter_moe_modules(model: Transformer) -> Iterable[MoEBase]:
    for block in model.blocks.values():
        if isinstance(block, MoETransformerBlock):
            yield block.feed_forward_moe


def set_moe_metrics_enabled(model: Transformer, enabled: bool) -> None:
    for moe_module in iter_moe_modules(model):
        moe_module.set_metrics_enabled(enabled)


def extract_batch_bytes(
    metadata: Optional[Sequence[object]],
    *,
    preferred_keys: Sequence[str],
) -> Optional[int]:
    if not metadata:
        return None

    def _extract(item) -> Optional[int]:
        if isinstance(item, dict):
            for key in preferred_keys:
                value = item.get(key)
                if isinstance(value, (int, float)):
                    return int(value)
            for value in item.values():
                result = _extract(value)
                if result is not None:
                    return result
        elif isinstance(item, list):
            for value in item:
                result = _extract(value)
                if result is not None:
                    return result
        return None

    total = 0
    found = False
    for entry in metadata:
        value = _extract(entry)
        if value is not None:
            total += value
            found = True
    return total if found else None
