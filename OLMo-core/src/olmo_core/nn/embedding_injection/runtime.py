from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

from olmo_core.exceptions import OLMoConfigurationError


@dataclass
class InjectionBlockContext:
    block_key: str
    block_idx: int
    step: Optional[int]
    input_ids: Optional[torch.Tensor]
    input_embedding: Optional[torch.Tensor]
    hidden_states: torch.Tensor


@dataclass
class InjectionBlockResult:
    hidden_states: torch.Tensor
    block_kwargs: Dict[str, Any] = field(default_factory=dict)


def resolve_configured_layers(
    configured_layers: Optional[List[int]],
    *,
    default_layers: Optional[List[int]] = None,
) -> List[int]:
    if configured_layers is not None:
        return list(configured_layers)
    return list(default_layers) if default_layers is not None else []


def validate_injection_layers(label: str, layers: List[int], *, n_layers: int) -> None:
    for layer_idx in layers:
        if layer_idx < 0 or layer_idx >= n_layers:
            raise OLMoConfigurationError(
                f"{label} layer index {layer_idx} is out of bounds for n_layers={n_layers}"
            )
