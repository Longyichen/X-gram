from typing import Any, Iterable, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from .runtime import InjectionBlockContext, InjectionBlockResult, resolve_configured_layers


def apply_mort_sparse_injection(
    hidden_states: Tensor,
    embeddings: Optional[Iterable[nn.Embedding]],
    weight_generator: Optional[nn.Module],
    scaler: Optional[torch.nn.Parameter],
    input_ids: Optional[Tensor],
    *,
    top_k: int,
    load_balancing_loss_weight: float,
    epsilon: float = 1e-6,
    depth_scale: float = 1.0,
) -> Tuple[Optional[Tensor], Optional[Tensor]]:
    """
    Top-K sparse variant of the MORT injection.
    """
    if (
        embeddings is None
        or weight_generator is None
        or scaler is None
        or input_ids is None
    ):
        return None, None

    embedding_list = list(embeddings)
    if not embedding_list:
        return None, None

    k = max(1, min(top_k, len(embedding_list)))
    logits = weight_generator(hidden_states)
    k = min(k, logits.shape[-1])
    if k < 1:
        return None, None

    topk_values, topk_indices = torch.topk(logits, k=k, dim=-1)
    weights = torch.sigmoid(topk_values)
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(epsilon)

    stacked_injections = torch.stack(
        [
            embedding_layer(input_ids).to(dtype=hidden_states.dtype, device=hidden_states.device)
            for embedding_layer in embedding_list
        ],
        dim=2,
    )

    weights_full = torch.zeros_like(logits)
    weights_full.scatter_(-1, topk_indices, weights)
    weighted = torch.einsum("bsn,bsnd->bsd", weights_full, stacked_injections)

    injection_fp32 = weighted.to(torch.float32)
    denom = torch.linalg.vector_norm(injection_fp32, dim=-1, keepdim=True).clamp_min(epsilon)
    normalized = injection_fp32 / denom
    scaler_view = scaler.to(torch.float32).view(1, 1, -1)
    scaled = normalized * scaler_view * depth_scale

    aux_loss: Optional[Tensor] = None
    batch_size, seq_len, num_experts = logits.shape
    total_tokens = batch_size * seq_len
    if num_experts > 0 and topk_indices.numel() > 0 and total_tokens > 0 and k > 0:
        logits_flat = logits.reshape(-1, num_experts)
        aggregate_probs = torch.sigmoid(logits_flat).sum(dim=0)

        flat_topk = topk_indices.reshape(-1)
        expert_counts = torch.zeros(
            num_experts,
            dtype=logits.dtype,
            device=logits.device,
        )
        expert_counts = expert_counts.scatter_add_(
            0,
            flat_topk,
            torch.ones_like(flat_topk, dtype=logits.dtype),
        ).detach()

        tokens_f = float(total_tokens)
        k_f = float(k)
        scaling = num_experts / (tokens_f * tokens_f * k_f)
        aux_loss = load_balancing_loss_weight * scaling * (aggregate_probs * expert_counts).sum()

    return scaled.to(dtype=hidden_states.dtype), aux_loss


def build_mort_modules(
    transformer: Any,
    embedding_injection: Any,
    *,
    vocab_size: int,
    d_model: int,
    dtype: torch.dtype,
    init_device: str,
    mort_aux_loss_weight: float,
) -> None:
    h_injection_layers = resolve_configured_layers(
        getattr(embedding_injection, "h_layers", None),
        default_layers=list(embedding_injection.layers),
    )

    for layer_idx in h_injection_layers:
        block_key = str(layer_idx)
        if block_key not in transformer._injection_h_embeddings:
            transformer._injection_h_embeddings[block_key] = nn.ModuleList()

        injection_embedding = nn.Embedding(
            vocab_size,
            d_model,
            dtype=dtype,
            device=init_device,
        )
        transformer._injection_h_embeddings[block_key].append(injection_embedding)

    mort_top_k = getattr(embedding_injection, "mort_top_k", None)
    if mort_top_k is None:
        mort_top_k = 2

    for block_key, embeddings in transformer._injection_h_embeddings.items():
        num_embeddings = len(embeddings)
        if num_embeddings == 0:
            continue
        block = transformer.blocks[block_key]
        block.mort_sparse_weight_generator = nn.Linear(
            d_model,
            num_embeddings,
            dtype=dtype,
            device=init_device,
        )
        block.mort_sparse_scaler = nn.Parameter(
            torch.empty(
                d_model,
                dtype=torch.float32,
                device=init_device,
            )
        )
        block.mort_sparse_aux_loss_weight = mort_aux_loss_weight
        block.mort_sparse_top_k = max(1, int(mort_top_k))


def prepare_mort_block_kwargs(
    transformer: Any,
    context: InjectionBlockContext,
) -> InjectionBlockResult:
    block_kwargs = {}
    if context.block_key in transformer._injection_h_embeddings:
        block_kwargs["_injection_h_embeddings"] = transformer._injection_h_embeddings[context.block_key]
        block_kwargs["input_ids"] = context.input_ids
    return InjectionBlockResult(hidden_states=context.hidden_states, block_kwargs=block_kwargs)


def init_mort_modules(
    transformer: Any,
    *,
    generator: torch.Generator,
) -> None:
    for block in transformer.blocks.values():
        if block.mort_sparse_weight_generator is not None:
            transformer.init_method._init_linear(
                block.mort_sparse_weight_generator,
                std=transformer.init_std,
                generator=generator,
            )
            if (
                block.mort_sparse_weight_generator.bias is not None
                and not block.mort_sparse_weight_generator.bias.is_meta
            ):
                nn.init.constant_(block.mort_sparse_weight_generator.bias, -2.0)
        if block.mort_sparse_scaler is not None and not block.mort_sparse_scaler.is_meta:
            nn.init.constant_(block.mort_sparse_scaler, 0.0)
