from typing import Any, Dict

import torch
import torch.nn as nn

from .runtime import InjectionBlockContext, InjectionBlockResult, resolve_configured_layers


def build_retoken_modules(
    transformer: Any,
    embedding_injection: Any,
    *,
    vocab_size: int,
    d_model: int,
    dtype: torch.dtype,
    init_device: str,
) -> None:
    del d_model

    h_injection_layers = resolve_configured_layers(
        getattr(embedding_injection, "h_layers", None),
        default_layers=list(embedding_injection.layers),
    )

    for layer_idx in h_injection_layers:
        block_key = str(layer_idx)
        if block_key not in transformer._retoken_embeddings:
            transformer._retoken_embeddings[block_key] = nn.ModuleList()
        if block_key not in transformer._retoken_scalers:
            transformer._retoken_scalers[block_key] = nn.ParameterList()

        injection_embedding = nn.Embedding(
            vocab_size,
            transformer.d_model,
            dtype=dtype,
            device=init_device,
        )
        transformer._retoken_embeddings[block_key].append(injection_embedding)
        scaler = nn.Parameter(
            torch.empty(
                transformer.d_model,
                dtype=torch.float32,
                device=init_device,
            )
        )
        transformer._retoken_scalers[block_key].append(scaler)


def prepare_retoken_block_kwargs(
    transformer: Any,
    context: InjectionBlockContext,
) -> InjectionBlockResult:
    block_kwargs: Dict[str, Any] = {}
    if (
        context.block_key in transformer._retoken_embeddings
        and context.block_key in transformer._retoken_scalers
    ):
        block_kwargs["_retoken_embeddings"] = transformer._retoken_embeddings[context.block_key]
        block_kwargs["_retoken_scalers"] = transformer._retoken_scalers[context.block_key]
        block_kwargs["input_ids"] = context.input_ids
    return InjectionBlockResult(hidden_states=context.hidden_states, block_kwargs=block_kwargs)


def init_retoken_modules(
    transformer: Any,
    *,
    generator: torch.Generator,
) -> None:
    for embeddings in transformer._retoken_embeddings.values():
        for embedding in embeddings:
            transformer.init_method.init_embeddings(
                embedding,
                d_model=transformer.d_model,
                std=transformer.init_std,
                generator=generator,
            )
    for scalers in transformer._retoken_scalers.values():
        for scaler in scalers:
            if not scaler.is_meta:
                nn.init.constant_(scaler, 0.0)
