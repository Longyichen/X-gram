import logging
from collections.abc import Callable
from dataclasses import dataclass
from fnmatch import fnmatch
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

import numpy as np

from olmo_core.config import Config, DType, StrEnum
from olmo_core.doc_utils import beta_feature
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.utils import ensure_multiple_of

from ..attention import AttentionConfig, AttentionType
from ..buffer_cache import BufferCache
from ..feed_forward import FeedForwardConfig, FeedForwardType
from ..layer_norm import LayerNormConfig, LayerNormType
from ..lm_head import LMHeadConfig, LMHeadType
from ..moe import MoEConfig, MoERouterConfig, MoEType
from ..rope import RoPEConfig, RoPEScalingConfig, RoPEType
from .init import InitMethod

if TYPE_CHECKING:
    from .block import TransformerBlockBase
    from .model import Transformer

log = logging.getLogger(__name__)

_XGRAM_TARGET_SET = {"h", "q", "k", "v", "o"}
_QKVO_TARGET_SET = {"q", "k", "v", "o"}


def _resolve_layers(
    configured_layers: Optional[List[int]],
    *,
    default_layers: Optional[List[int]] = None,
) -> List[int]:
    if configured_layers is not None:
        return list(configured_layers)
    return list(default_layers) if default_layers is not None else []


@lru_cache(maxsize=None)
def _load_hash_token_map_total_capacity(token_map_path: str) -> int:
    path = Path(token_map_path)
    if not path.exists():
        raise OLMoConfigurationError(f"hash token map '{token_map_path}' does not exist")
    with np.load(path, allow_pickle=True) as data:
        if "total_capacity" not in data:
            raise OLMoConfigurationError(
                f"hash token map '{token_map_path}' is missing required field 'total_capacity'"
            )
        return int(np.asarray(data["total_capacity"]).reshape(()).item())


def _attention_injection_dim(d_model: int, attention: AttentionConfig) -> int:
    n_heads = attention.n_heads
    n_kv_heads = attention.n_kv_heads or n_heads
    if n_heads <= 0 or n_kv_heads <= 0 or d_model % n_heads != 0:
        raise OLMoConfigurationError(
            f"Invalid attention config for X-gram param counting: d_model={d_model}, "
            f"n_heads={n_heads}, n_kv_heads={n_kv_heads}"
        )
    return n_kv_heads * (d_model // n_heads)


def _swiglu_shortconv_num_params(dim: int, kernel_size: int) -> int:
    return dim * (2 * kernel_size + 1)


def _is_prime(n: int) -> bool:
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


def _next_prime(start: int, seen: Set[int]) -> int:
    candidate = max(2, start)
    while True:
        if _is_prime(candidate) and candidate not in seen:
            return candidate
        candidate += 1



class TransformerDataParallelWrappingStrategy(StrEnum):
    """
    An enumeration of the different wrapping strategy for the data parallel implementations.
    """

    full = "full"
    """
    Wrap each block and the LM head (only applies to FSDP).
    """

    blocks = "blocks"
    """
    Like full but the LM head is not wrapped separately (only applies to FSDP).
    """

    fine_grained = "fine_grained"
    """
    Wrap certain modules within each block in addition to wrapping each block (only applies to FSDP).
    """


@beta_feature
class TransformerActivationCheckpointingMode(StrEnum):
    """
    An enumeration of the different activation checkpointing modes.
    """

    full = "full"
    """Checkpoint every block."""
    selected_blocks = "selected_blocks"
    """Checkpoint only selected blocks."""
    selected_modules = "selected_modules"
    """Checkpoint only selected modules."""
    selected_ops = "selected_ops"
    """Checkpoint only a specific set of operations."""
    budget = "budget"
    """Checkpoint based on a budget."""


class TransformerType(StrEnum):
    """
    An enumeration of transformer implementations.
    """

    default = "default"
    """
    ➡️ :class:`Transformer`
    """

    normalized = "normalized"
    """
    ➡️ :class:`NormalizedTransformer` (nGPT)
    """

    moe = "moe"
    """
    ➡️ :class:`MoETransformer`
    """


class TransformerBlockType(StrEnum):
    """
    An enumeration of the different transformer block implementations.
    """

    default = "default"
    """
    ➡️ :class:`TransformerBlock`
    """

    reordered_norm = "reordered_norm"
    """
    ➡️ :class:`ReorderedNormTransformerBlock`
    """

    normalized = "normalized"
    """
    ➡️ :class:`NormalizedTransformerBlock`
    """

    moe = "moe"
    """
    ➡️ :class:`MoETransformerBlock`
    """

    moe_reordered_norm = "moe_reordered_norm"
    """
    ➡️ :class:`MoEReorderedNormTransformerBlock`
    """

    moe_hybrid = "moe_hybrid"
    """
    ➡️ :class:`MoEHybridTransformerBlock`
    """

    moe_hybrid_reordered_norm = "moe_hybrid_reordered_norm"
    """
    ➡️ :class:`MoEHybridReorderedNormTransformerBlock`
    """


@dataclass
class TransformerEmbeddingInjectionConfig(Config):
    """
    Configure the strategy for inserting extra embeddings inside Transformer blocks.

    - ``layers`` is the list of layer indices for all injection module instances
      and may contain duplicates.
    - ``h_layers/qk_layers/q_layers/k_layers/v_layers/o_layers`` split X-gram
      injection positions by target.
    - ``hash_*`` are hash-layout parameters used by X-gram hash injection.
    """
    layers: List[int]
    h_layers: Optional[List[int]] = None
    qk_layers: Optional[List[int]] = None
    q_layers: Optional[List[int]] = None
    k_layers: Optional[List[int]] = None
    v_layers: Optional[List[int]] = None
    o_layers: Optional[List[int]] = None
    mode: Optional[str] = None
    targets: Optional[List[str]] = None
    qk_sharing: bool = False
    shortconv_enabled: bool = False
    shortconv_kernels: Optional[List[int]] = None
    hash_enabled: bool = False
    hash_token_map_path: Optional[str] = None
    lambda_init: float = 1.0
    lambda_warmup_enabled: bool = True
    lambda_warmup_steps: int = 0
    log_interval: int = 0
    depth_scale_disabled: bool = False
    engram_tokenizer_id: Optional[str] = None
    engram_cache_path: Optional[str] = None
    engram_heads: int = 8
    engram_target_buckets: Optional[int] = 75968
    engram_reduction: float = 0.5
    engram_base_seed: int = 42
    engram_hc_mult: int = 1
    engram_shortconv_enabled: bool = True
    engram_shortconv_kernel: int = 4
    engram_shortconv_dilation: int = 1
    engram_shortconv_activation: bool = True
    engram_mode: str = "2gram+3gram"
    engram_dim_per_ngram: Optional[int] = None
    engram_ngram_heads: int = 8
    engram_ngram_target_buckets: Optional[int] = 75968
    engram_ngram_seed: int = 137
    mort_top_k: Optional[int] = None

    def __post_init__(self):
        layer_fields = {
            "layers": self.layers,
            "h_layers": self.h_layers,
            "qk_layers": self.qk_layers,
            "q_layers": self.q_layers,
            "k_layers": self.k_layers,
            "v_layers": self.v_layers,
            "o_layers": self.o_layers,
        }
        if not any(layer_values for layer_values in layer_fields.values() if layer_values is not None):
            raise OLMoConfigurationError(
                "TransformerEmbeddingInjectionConfig requires at least one non-empty layer list"
            )
        for field_name, layer_values in layer_fields.items():
            if layer_values is None:
                continue
            for layer_idx in layer_values:
                if layer_idx < 0:
                    raise OLMoConfigurationError(
                        f"TransformerEmbeddingInjectionConfig.{field_name} must be non-negative, got {layer_idx}"
                    )
        if self.targets is not None:
            normalized_targets: List[str] = []
            invalid_targets: List[str] = []
            for token in self.targets:
                target = str(token).strip().lower()
                if target in {"h", "q", "k", "v", "o"}:
                    if target not in normalized_targets:
                        normalized_targets.append(target)
                else:
                    invalid_targets.append(str(token))
            if invalid_targets:
                raise OLMoConfigurationError(
                    "TransformerEmbeddingInjectionConfig.targets contains invalid entries: "
                    f"{invalid_targets}"
                )
            self.targets = normalized_targets
        mode = str(self.mode or "None").strip()
        if mode in {"Retoken", "Mort", "Engram"}:
            if not self.h_layers:
                raise OLMoConfigurationError(
                    f"TransformerEmbeddingInjectionConfig.mode='{mode}' requires a non-empty h_layers. "
                    "These modes do not fall back to the combined layers field."
                )
            unexpected_layer_fields = [
                field_name
                for field_name in ("qk_layers", "q_layers", "k_layers", "v_layers", "o_layers")
                if getattr(self, field_name)
            ]
            if unexpected_layer_fields:
                raise OLMoConfigurationError(
                    f"TransformerEmbeddingInjectionConfig.mode='{mode}' only supports h_layers; "
                    f"unexpected non-empty layer fields: {unexpected_layer_fields}"
                )
        if self.qk_sharing and (
            (self.q_layers is not None and len(self.q_layers) > 0)
            or (self.k_layers is not None and len(self.k_layers) > 0)
        ):
            raise OLMoConfigurationError(
                "TransformerEmbeddingInjectionConfig.qk_sharing=True does not allow q_layers or k_layers; use qk_layers instead"
            )
        if self.shortconv_kernels is not None:
            normalized_kernels: List[int] = []
            for kernel in self.shortconv_kernels:
                kernel_int = int(kernel)
                if kernel_int <= 0:
                    raise OLMoConfigurationError(
                        "TransformerEmbeddingInjectionConfig.shortconv_kernels must contain positive integers"
                    )
                normalized_kernels.append(kernel_int)
            self.shortconv_kernels = normalized_kernels
        if self.lambda_warmup_steps < 0:
            raise OLMoConfigurationError(
                "TransformerEmbeddingInjectionConfig.lambda_warmup_steps must be non-negative"
            )
        if self.log_interval < 0:
            raise OLMoConfigurationError(
                "TransformerEmbeddingInjectionConfig.log_interval must be non-negative"
            )
        if self.hash_enabled and not self.hash_token_map_path:
            raise OLMoConfigurationError(
                "TransformerEmbeddingInjectionConfig.hash_enabled=True requires hash_token_map_path"
            )
        if self.engram_heads <= 0:
            raise OLMoConfigurationError("TransformerEmbeddingInjectionConfig.engram_heads must be positive")
        if self.engram_target_buckets is not None and self.engram_target_buckets <= 0:
            raise OLMoConfigurationError(
                "TransformerEmbeddingInjectionConfig.engram_target_buckets must be positive"
            )
        if self.engram_reduction <= 0:
            raise OLMoConfigurationError(
                "TransformerEmbeddingInjectionConfig.engram_reduction must be positive"
            )
        if self.engram_hc_mult <= 0:
            raise OLMoConfigurationError("TransformerEmbeddingInjectionConfig.engram_hc_mult must be positive")
        if self.engram_shortconv_kernel <= 0:
            raise OLMoConfigurationError(
                "TransformerEmbeddingInjectionConfig.engram_shortconv_kernel must be positive"
            )
        if self.engram_shortconv_dilation <= 0:
            raise OLMoConfigurationError(
                "TransformerEmbeddingInjectionConfig.engram_shortconv_dilation must be positive"
            )
        if self.engram_dim_per_ngram is not None and self.engram_dim_per_ngram <= 0:
            raise OLMoConfigurationError(
                "TransformerEmbeddingInjectionConfig.engram_dim_per_ngram must be positive"
            )
        if self.engram_ngram_heads <= 0:
            raise OLMoConfigurationError(
                "TransformerEmbeddingInjectionConfig.engram_ngram_heads must be positive"
            )
        if self.engram_ngram_target_buckets is not None and self.engram_ngram_target_buckets <= 0:
            raise OLMoConfigurationError(
                "TransformerEmbeddingInjectionConfig.engram_ngram_target_buckets must be positive"
            )
        if self.mort_top_k is not None and self.mort_top_k <= 0:
            raise OLMoConfigurationError("TransformerEmbeddingInjectionConfig.mort_top_k must be positive")


@dataclass
class TransformerBlockConfig(Config):
    """
    A configuration class for easily building transformer blocks.
    """

    attention: AttentionConfig
    """
    The attention config.
    """
    layer_norm: Optional[LayerNormConfig] = None
    """
    The layer norm config.
    """
    feed_forward: Optional[FeedForwardConfig] = None
    """
    The feed-forward config, required for non-MoE blocks.
    """
    feed_forward_moe: Optional[MoEConfig] = None
    """
    The config for the MoE feed-forward layer. Required for MoE blocks.
    """
    name: TransformerBlockType = TransformerBlockType.default
    """
    The block type.
    """
    dropout: Optional[float] = None
    """
    Dropout probability.
    """

    def build(
        self,
        *,
        d_model: int,
        block_idx: int,
        n_layers: int,
        init_device: str = "cpu",
        cache: Optional[BufferCache] = None,
    ) -> "TransformerBlockBase":
        from .block import (
            MoEHybridReorderedNormTransformerBlock,
            MoEHybridTransformerBlock,
            MoEReorderedNormTransformerBlock,
            MoETransformerBlock,
            NormalizedTransformerBlock,
            ReorderedNormTransformerBlock,
            TransformerBlock,
        )

        kwargs = self.as_dict(exclude_none=True, recurse=False)
        kwargs.pop("name")
        kwargs.update(
            d_model=d_model,
            block_idx=block_idx,
            n_layers=n_layers,
            init_device=init_device,
            cache=cache,
        )

        try:
            if self.name == TransformerBlockType.default:
                return TransformerBlock(**kwargs)
            elif self.name == TransformerBlockType.reordered_norm:
                return ReorderedNormTransformerBlock(**kwargs)
            elif self.name == TransformerBlockType.normalized:
                return NormalizedTransformerBlock(**kwargs)
            elif self.name == TransformerBlockType.moe:
                return MoETransformerBlock(**kwargs)
            elif self.name == TransformerBlockType.moe_reordered_norm:
                return MoEReorderedNormTransformerBlock(**kwargs)
            elif self.name == TransformerBlockType.moe_hybrid:
                return MoEHybridTransformerBlock(**kwargs)
            elif self.name == TransformerBlockType.moe_hybrid_reordered_norm:
                return MoEHybridReorderedNormTransformerBlock(**kwargs)
            else:
                raise NotImplementedError(self.name)
        except TypeError as e:
            raise OLMoConfigurationError(
                f"invalid options for '{self.name}' {self.__class__.__name__}, {e}"
            ) from e

    def num_params(self, d_model: int) -> int:
        block_params = 0

        # Block attn and MLP scaling factors.
        if self.name == TransformerBlockType.normalized:
            block_params += 2 * d_model

        # Block attention params.
        block_params += self.attention.num_params(d_model)
        if self.layer_norm is not None:
            block_params += self.layer_norm.num_params(d_model)

        # Block feed forward (dense and/or sparse).
        if self.feed_forward is not None:
            block_params += self.feed_forward.num_params(d_model)
            if self.layer_norm is not None:
                block_params += self.layer_norm.num_params(d_model)
        if self.feed_forward_moe is not None:
            block_params += self.feed_forward_moe.num_params(d_model)
            if self.layer_norm is not None:
                block_params += self.layer_norm.num_params(d_model)

        return block_params

    def num_active_params(self, d_model: int) -> int:
        num_params = self.num_params(d_model)
        if self.feed_forward_moe is None:
            return num_params

        num_inactive_params = self.feed_forward_moe.num_params(
            d_model
        ) - self.feed_forward_moe.num_active_params(d_model)
        return num_params - num_inactive_params


@dataclass
class TransformerConfig(Config):
    """
    A config for easily building transformer models.

    :param name: The name of the implementation.

    See :class:`Transformer` for a description of the other parameters.
    """

    d_model: int
    vocab_size: int
    n_layers: int
    block: TransformerBlockConfig
    lm_head: LMHeadConfig
    name: TransformerType = TransformerType.default
    dtype: DType = DType.float32
    init_method: InitMethod = InitMethod.normal
    init_seed: int = 0
    init_std: float = 0.02
    freeze_params: Optional[List[str]] = None
    block_overrides: Optional[Dict[int, TransformerBlockConfig]] = None
    embedding_injection: Optional[TransformerEmbeddingInjectionConfig] = None

    def build(
        self,
        *,
        init_device: str = "cpu",
    ) -> "Transformer":
        """
        Build the model corresponding to this config.

        :param init_device: The device to put the parameters on during initialization. In a
            distributed setting it usually makes sense to set this to "meta".
        """
        from .model import MoETransformer, NormalizedTransformer, Transformer

        model: Transformer
        if self.name == TransformerType.default:
            model = Transformer(
                d_model=self.d_model,
                vocab_size=self.vocab_size,
                n_layers=self.n_layers,
                block=self.block,
                lm_head=self.lm_head,
                dtype=self.dtype.as_pt(),
                init_method=self.init_method,
                init_device=init_device,
                init_seed=self.init_seed,
                init_std=self.init_std,
                block_overrides=self.block_overrides,
                embedding_injection=self.embedding_injection,
            )
        elif self.name == TransformerType.normalized:
            model = NormalizedTransformer(
                d_model=self.d_model,
                vocab_size=self.vocab_size,
                n_layers=self.n_layers,
                block=self.block,
                lm_head=self.lm_head,
                dtype=self.dtype.as_pt(),
                init_method=self.init_method,
                init_device=init_device,
                init_seed=self.init_seed,
                init_std=self.init_std,
                block_overrides=self.block_overrides,
                embedding_injection=self.embedding_injection,
            )
        elif self.name == TransformerType.moe:
            model = MoETransformer(
                d_model=self.d_model,
                vocab_size=self.vocab_size,
                n_layers=self.n_layers,
                block=self.block,
                lm_head=self.lm_head,
                dtype=self.dtype.as_pt(),
                init_method=self.init_method,
                init_device=init_device,
                init_seed=self.init_seed,
                init_std=self.init_std,
                block_overrides=self.block_overrides,
                embedding_injection=self.embedding_injection,
            )
        else:
            raise NotImplementedError(self.name)

        non_embed_params = getattr(model, "num_non_embedding_params", None)
        if non_embed_params is None:
            log.warning("Model is missing 'num_non_embedding_params'; reporting 0 for logging.")
            non_embed_params = 0
        log.info(
            f"Building transformer with {model.num_params:,d} total params, "
            f"{non_embed_params:,d} non-embedding params"
        )

        if self.freeze_params:
            for name, param in model.named_parameters():
                for pattern in self.freeze_params:
                    if fnmatch(name, pattern):
                        param.requires_grad = False
                        log.info(f"Param '{name}' will be frozen")
                        break
                else:
                    log.info(f"Param '{name}' will be trainable")

        log.info("%s", model)
        log.info(
            f"Built model with:\n"
            f"- {model.num_params:,d} total params\n"
            f"- {non_embed_params:,d} non-embedding params\n"
            f"- {model.num_trainable_params:,d} trainable params"
        )

        return model

    def _count_xgram_container_params(
        self,
        layers: List[int],
        *,
        dim: int,
        hash_enabled: bool,
        shortconv_enabled: bool,
        kernels: List[int],
        hash_total_capacity: Optional[int],
    ) -> Tuple[int, int]:
        module_count = len(layers)
        if module_count == 0:
            return 0, 0

        if hash_enabled:
            if hash_total_capacity is None:
                raise OLMoConfigurationError(
                    "X-gram hash parameter counting requires hash_token_map_path to be set"
                )
            # Hash injection has one bucket embedding table plus one scalar-weight embedding
            # table per module.
            embedding_like = module_count * hash_total_capacity * (dim + 1)
        else:
            embedding_like = module_count * self.vocab_size * dim

        total = embedding_like + module_count  # one scalar gate parameter per module
        if shortconv_enabled:
            per_block_counts: Dict[int, int] = {}
            for layer_idx in layers:
                conv_idx = per_block_counts.get(layer_idx, 0)
                kernel_size = kernels[conv_idx % len(kernels)]
                total += _swiglu_shortconv_num_params(dim, kernel_size)
                per_block_counts[layer_idx] = conv_idx + 1

        return total, embedding_like

    def _count_xgram_container_params_per_layer(
        self,
        layers: List[int],
        *,
        dim_by_layer: Dict[int, int],
        hash_enabled: bool,
        shortconv_enabled: bool,
        kernels: List[int],
        hash_total_capacity: Optional[int],
    ) -> Tuple[int, int]:
        total = 0
        embedding_like = 0
        if not layers:
            return total, embedding_like

        per_block_counts: Dict[int, int] = {}
        for layer_idx in layers:
            dim = dim_by_layer[layer_idx]
            if hash_enabled:
                if hash_total_capacity is None:
                    raise OLMoConfigurationError(
                        "X-gram hash parameter counting requires hash_token_map_path to be set"
                    )
                embedding_like += hash_total_capacity * (dim + 1)
            else:
                embedding_like += self.vocab_size * dim

            total += 1  # scalar gate parameter
            if shortconv_enabled:
                conv_idx = per_block_counts.get(layer_idx, 0)
                kernel_size = kernels[conv_idx % len(kernels)]
                total += _swiglu_shortconv_num_params(dim, kernel_size)
                per_block_counts[layer_idx] = conv_idx + 1

        total += embedding_like
        return total, embedding_like

    def _block_config_for_layer(self, layer_idx: int) -> TransformerBlockConfig:
        if self.block_overrides is not None and layer_idx in self.block_overrides:
            return self.block_overrides[layer_idx]
        return self.block

    def _xgram_injection_param_counts(self) -> Tuple[int, int]:
        cfg = self.embedding_injection
        if cfg is None:
            return 0, 0

        injection_targets = list(cfg.targets or ["h"])
        attention_targets = [target for target in injection_targets if target in _QKVO_TARGET_SET]
        h_target_enabled = "h" in injection_targets
        attention_qkv_targets = [target for target in attention_targets if target in {"q", "k", "v"}]
        attention_qkv_target_set = set(attention_qkv_targets)
        attention_qk_sharing_active = cfg.qk_sharing and {"q", "k"}.issubset(attention_qkv_target_set)

        default_h_layers = _resolve_layers(getattr(cfg, "h_layers", None), default_layers=list(cfg.layers))
        default_qkv_layers = default_h_layers
        configured_qk_layers = list(cfg.qk_layers) if cfg.qk_layers is not None else None
        target_layers = {
            "q": _resolve_layers(
                configured_qk_layers if attention_qk_sharing_active else getattr(cfg, "q_layers", None),
                default_layers=default_qkv_layers if "q" in attention_targets else [],
            ),
            "k": _resolve_layers(
                configured_qk_layers if attention_qk_sharing_active else getattr(cfg, "k_layers", None),
                default_layers=default_qkv_layers if "k" in attention_targets else [],
            ),
            "v": _resolve_layers(
                getattr(cfg, "v_layers", None),
                default_layers=default_qkv_layers if "v" in attention_targets else [],
            ),
            "o": _resolve_layers(
                getattr(cfg, "o_layers", None),
                default_layers=default_h_layers if "o" in attention_targets else [],
            ),
        }

        hash_enabled = bool(getattr(cfg, "hash_enabled", False))
        hash_total_capacity: Optional[int] = None
        if hash_enabled:
            token_map_path = getattr(cfg, "hash_token_map_path", None)
            if not token_map_path:
                raise OLMoConfigurationError(
                    "TransformerEmbeddingInjectionConfig.hash_enabled=True requires hash_token_map_path"
                )
            hash_total_capacity = _load_hash_token_map_total_capacity(str(token_map_path))

        shortconv_enabled = bool(getattr(cfg, "shortconv_enabled", False))
        kernels = list(getattr(cfg, "shortconv_kernels", None) or [3, 5, 7, 9])
        attention_dim_by_layer = {
            layer_idx: _attention_injection_dim(
                self.d_model,
                self._block_config_for_layer(layer_idx).attention,
            )
            for layer_idx in set(target_layers["q"] + target_layers["k"] + target_layers["v"])
        }

        total = 0
        embedding_like = 0
        if h_target_enabled:
            h_total, h_embedding_like = self._count_xgram_container_params(
                default_h_layers,
                dim=self.d_model,
                hash_enabled=hash_enabled,
                shortconv_enabled=shortconv_enabled,
                kernels=kernels,
                hash_total_capacity=hash_total_capacity,
            )
            total += h_total
            embedding_like += h_embedding_like

        qkv_specs: List[Tuple[str, List[int]]] = []
        if attention_qk_sharing_active:
            qkv_specs.append(("qk", target_layers["q"]))
        else:
            for target in attention_qkv_targets:
                qkv_specs.append((target, target_layers[target]))
        if "v" in attention_qkv_target_set and all(label != "v" for label, _ in qkv_specs):
            qkv_specs.append(("v", target_layers["v"]))

        for _, layers in qkv_specs:
            qkv_total, qkv_embedding_like = self._count_xgram_container_params_per_layer(
                layers,
                dim_by_layer=attention_dim_by_layer,
                hash_enabled=hash_enabled,
                shortconv_enabled=shortconv_enabled,
                kernels=kernels,
                hash_total_capacity=hash_total_capacity,
            )
            total += qkv_total
            embedding_like += qkv_embedding_like

        o_total, o_embedding_like = self._count_xgram_container_params(
            target_layers["o"],
            dim=self.d_model,
            hash_enabled=hash_enabled,
            shortconv_enabled=shortconv_enabled,
            kernels=kernels,
            hash_total_capacity=hash_total_capacity,
        )
        total += o_total
        embedding_like += o_embedding_like

        return total, embedding_like

    def _retoken_injection_param_counts(self) -> Tuple[int, int]:
        cfg = self.embedding_injection
        if cfg is None:
            return 0, 0
        h_layers = _resolve_layers(getattr(cfg, "h_layers", None), default_layers=list(cfg.layers))
        module_count = len(h_layers)
        embedding_like = module_count * self.d_model * self.vocab_size
        total = embedding_like + module_count * self.d_model
        return total, embedding_like

    def _mort_injection_param_counts(self) -> Tuple[int, int]:
        cfg = self.embedding_injection
        if cfg is None:
            return 0, 0
        h_layers = _resolve_layers(getattr(cfg, "h_layers", None), default_layers=list(cfg.layers))
        module_count = len(h_layers)
        embedding_like = module_count * self.d_model * self.vocab_size
        total = embedding_like
        per_block_counts: Dict[int, int] = {}
        for layer_idx in h_layers:
            per_block_counts[layer_idx] = per_block_counts.get(layer_idx, 0) + 1
        for num_embeddings in per_block_counts.values():
            total += self.d_model * num_embeddings  # weight generator weight
            total += num_embeddings  # weight generator bias
            total += self.d_model  # mort sparse scaler
        return total, embedding_like

    def _engram_injection_param_counts(self) -> Tuple[int, int]:
        cfg = self.embedding_injection
        if cfg is None:
            return 0, 0

        h_layers = _resolve_layers(getattr(cfg, "h_layers", None), default_layers=list(cfg.layers))
        if not h_layers:
            return 0, 0
        if len(set(h_layers)) != len(h_layers):
            raise OLMoConfigurationError(
                "Engram supports at most one module per layer; duplicate layer indices were provided"
            )

        engram_mode = str(getattr(cfg, "engram_mode", "2gram+3gram"))
        mode_parts = {token.strip() for token in engram_mode.split("+") if token.strip()}
        ngram_levels = sorted(int(part.replace("gram", "")) for part in mode_parts)
        if len(ngram_levels) == 0:
            raise OLMoConfigurationError("Engram mode requires at least one n-gram level")
        if any(level < 2 for level in ngram_levels):
            raise OLMoConfigurationError("Engram mode only supports n-gram levels >= 2")

        hc_mult = int(getattr(cfg, "engram_hc_mult", 1))
        ngram_heads = int(getattr(cfg, "engram_ngram_heads", 4))
        target_capacity = int(getattr(cfg, "engram_ngram_target_buckets", 75968))
        dim_per_ngram_cfg = getattr(cfg, "engram_dim_per_ngram", None)
        dim_per_level = int(dim_per_ngram_cfg) if dim_per_ngram_cfg is not None else self.d_model // len(ngram_levels)

        if dim_per_level < 1:
            raise OLMoConfigurationError("ENGRAM_NGRAM_DIM must be >= 1")
        if ngram_heads < 1:
            raise OLMoConfigurationError("ENGRAM_NGRAM_HEADS must be >= 1")
        if dim_per_level % ngram_heads != 0:
            raise OLMoConfigurationError(
                f"ENGRAM_NGRAM_DIM ({dim_per_level}) must be divisible by heads ({ngram_heads})"
            )
        if target_capacity < 1:
            raise OLMoConfigurationError("ENGRAM_NGRAM_TARGET_BUCKETS must be > 0")

        engram_hidden_size = len(ngram_levels) * dim_per_level
        total = 0
        embedding_like = 0
        seen_primes: Set[int] = set()
        per_head_capacity = max(2, max(ngram_heads, target_capacity) // ngram_heads)
        align = 16
        for _layer_idx in h_layers:
            embedding_like_per_module = 0
            for _ in ngram_levels:
                current = per_head_capacity
                primes: List[int] = []
                for _ in range(ngram_heads):
                    prime = _next_prime(current, seen_primes)
                    primes.append(prime)
                    seen_primes.add(prime)
                    current = prime + 1
                total_embeddings = ((sum(primes) + align - 1) // align) * align
                embedding_like_per_module += total_embeddings * (dim_per_level // ngram_heads)

            total_per_module = embedding_like_per_module
            total_per_module += engram_hidden_size * self.d_model  # value_proj
            total_per_module += hc_mult * engram_hidden_size * self.d_model  # key_projs
            total_per_module += 2 * hc_mult * self.d_model  # norm1 + norm2
            if bool(getattr(cfg, "engram_shortconv_enabled", True)):
                kernel_size = int(getattr(cfg, "engram_shortconv_kernel", 4))
                total_per_module += self.d_model * hc_mult * kernel_size  # depthwise conv
                total_per_module += hc_mult * self.d_model  # per-hc RMSNorm weights

            total += total_per_module
            embedding_like += embedding_like_per_module

        return total, embedding_like

    def _fallback_injection_param_counts(self) -> Tuple[int, int]:
        cfg = self.embedding_injection
        if cfg is None:
            return 0, 0
        embedding_like = len(cfg.layers) * self.d_model * self.vocab_size
        total = embedding_like
        if (cfg.mode or "None") == "Retoken":
            total += len(cfg.layers) * self.d_model
        return total, embedding_like

    def _injection_param_counts(self) -> Tuple[int, int]:
        cfg = self.embedding_injection
        if cfg is None:
            return 0, 0
        mode = str(cfg.mode or "None").strip()
        if mode == "X-gram":
            return self._xgram_injection_param_counts()
        if mode == "Retoken":
            return self._retoken_injection_param_counts()
        if mode == "Mort":
            return self._mort_injection_param_counts()
        if mode == "Engram":
            return self._engram_injection_param_counts()
        return self._fallback_injection_param_counts()

    @property
    def num_params(self) -> int:
        """
        The total number of parameters that a model from this config would have.
        """
        num_params = 0

        # Embedding params.
        num_params += self.d_model * self.vocab_size
        injection_total_params, _ = self._injection_param_counts()
        num_params += injection_total_params

        # All block params.
        num_block_params = self.block.num_params(self.d_model)
        if self.block_overrides is None:
            num_params += self.n_layers * num_block_params
        else:
            for idx in range(self.n_layers):
                if idx in self.block_overrides:
                    num_params += self.block_overrides[idx].num_params(self.d_model)
                else:
                    num_params += num_block_params

        # LM head.
        num_params += self.lm_head.num_params(self.d_model, self.vocab_size)

        return num_params

    @property
    def num_active_params(self) -> int:
        """
        The total number of active parameters that a model from this config would have.
        """
        num_active_params = 0

        # Embedding params.
        num_active_params += self.d_model * self.vocab_size
        injection_total_params, _ = self._injection_param_counts()
        num_active_params += injection_total_params

        # All block active params.
        num_active_block_params = self.block.num_active_params(self.d_model)
        if self.block_overrides is None:
            num_active_params += self.n_layers * num_active_block_params
        else:
            for idx in range(self.n_layers):
                if idx in self.block_overrides:
                    num_active_params += self.block_overrides[idx].num_active_params(self.d_model)
                else:
                    num_active_params += num_active_block_params

        # LM head.
        num_active_params += self.lm_head.num_params(self.d_model, self.vocab_size)

        return num_active_params

    @property
    def num_non_embedding_params(self) -> int:
        """
        The number of parameters excluding embedding parameters.
        """
        _, injection_embedding_like_params = self._injection_param_counts()
        return self.num_params - self.d_model * self.vocab_size - injection_embedding_like_params

    @property
    def num_active_non_embedding_params(self) -> int:
        """
        The number of active parameters excluding embedding parameters.
        """
        _, injection_embedding_like_params = self._injection_param_counts()
        return self.num_active_params - self.d_model * self.vocab_size - injection_embedding_like_params

    def num_flops_per_token(self, seq_len: int) -> int:
        """
        Get the approximate number of flops per token.
        """
        n, h, q, t = (
            self.n_layers,
            self.block.attention.n_heads,
            self.d_model // self.block.attention.n_heads,
            seq_len,
        )
        # Reasoning behind the factor of 12 for the self-attention part of the formula:
        # 1. each self-attention has 2 matmul in the forward and 4 in the backward (6)
        # 2. the flash attention does 1 more matmul recomputation in the backward
        #    but recomputation should not be counted in calculating MFU           (+0)
        # 3. each matmul performs 1 multiplication and 1 addition                 (*2)
        # 4. we follow the convention and do not account for sparsity in causal attention
        flop_per_token = 6 * self.num_non_embedding_params + 12 * n * h * q * t

        return flop_per_token

    @classmethod
    def olmo2_190M(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        return cls.llama_like(
            d_model=768,
            hidden_size_multiplier=1.5,
            n_layers=kwargs.pop("n_layers", 12),
            n_heads=kwargs.pop("n_heads", 12),
            vocab_size=vocab_size,
            block_name=kwargs.pop("block_name", TransformerBlockType.reordered_norm),
            qk_norm=kwargs.pop("qk_norm", True),
            rope_theta=kwargs.pop("rope_theta", 500_000),
            layer_norm_eps=1e-6,
            **kwargs,
        )

    @classmethod
    def olmo2_370M(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        return cls.llama_like(
            d_model=1024,
            hidden_size_multiplier=1.5,
            n_layers=kwargs.pop("n_layers", 16),
            n_heads=kwargs.pop("n_heads", 16),
            vocab_size=vocab_size,
            block_name=kwargs.pop("block_name", TransformerBlockType.reordered_norm),
            qk_norm=kwargs.pop("qk_norm", True),
            rope_theta=kwargs.pop("rope_theta", 500_000),
            layer_norm_eps=1e-6,
            **kwargs,
        )

    @classmethod
    def olmo2_600M(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        return cls.llama_like(
            d_model=1344,
            hidden_size_multiplier=1.5,
            n_layers=kwargs.pop("n_layers", 16),
            n_heads=kwargs.pop("n_heads", 16),
            vocab_size=vocab_size,
            block_name=kwargs.pop("block_name", TransformerBlockType.reordered_norm),
            qk_norm=kwargs.pop("qk_norm", True),
            rope_theta=kwargs.pop("rope_theta", 500_000),
            layer_norm_eps=1e-6,
            **kwargs,
        )

    @classmethod
    def olmo2_760M(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        return cls.llama_like(
            d_model=1536,
            hidden_size_multiplier=1.5,
            n_layers=kwargs.pop("n_layers", 16),
            n_heads=kwargs.pop("n_heads", 16),
            vocab_size=vocab_size,
            block_name=kwargs.pop("block_name", TransformerBlockType.reordered_norm),
            qk_norm=kwargs.pop("qk_norm", True),
            rope_theta=kwargs.pop("rope_theta", 500_000),
            layer_norm_eps=1e-6,
            **kwargs,
        )

    @classmethod
    def olmo2_1B(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        """
        A 1B OLMo model config.

        This is different from the OLMo 1B from the old OLMo trainer.
        """
        return cls.llama2_1B(
            vocab_size,
            block_name=kwargs.pop("block_name", TransformerBlockType.reordered_norm),
            qk_norm=kwargs.pop("qk_norm", True),
            rope_theta=kwargs.pop("rope_theta", 500_000),
            layer_norm_eps=1e-6,
            hidden_size_multiplier=1.5,
            **kwargs,
        )

    @classmethod
    def olmo2_1B_v2(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        """
        A 1B OLMo model config.

        This matches the OLMo 1B from the old OLMo trainer.
        """
        return cls.llama2_1B(
            vocab_size,
            block_name=kwargs.pop("block_name", TransformerBlockType.reordered_norm),
            qk_norm=kwargs.pop("qk_norm", True),
            rope_theta=kwargs.pop("rope_theta", 500_000),
            layer_norm_eps=1e-6,
            n_layers=kwargs.pop("n_layers", 16),
            hidden_size_multiplier=kwargs.pop("hidden_size_multiplier", 1.5),
            **kwargs,
        )

    @classmethod
    def olmo2_3B(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        return cls.llama_like(
            d_model=3328,
            hidden_size_multiplier=1.5,
            n_layers=kwargs.pop("n_layers", 16),
            n_heads=kwargs.pop("n_heads", 16),
            vocab_size=vocab_size,
            block_name=kwargs.pop("block_name", TransformerBlockType.reordered_norm),
            qk_norm=kwargs.pop("qk_norm", True),
            rope_theta=kwargs.pop("rope_theta", 500_000),
            layer_norm_eps=1e-6,
            **kwargs,
        )

    @classmethod
    def olmo2_7B(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        """
        A 7B OLMo model config.
        """
        return cls.llama2_7B(
            vocab_size,
            block_name=kwargs.pop("block_name", TransformerBlockType.reordered_norm),
            qk_norm=kwargs.pop("qk_norm", True),
            rope_theta=kwargs.pop("rope_theta", 500_000),
            layer_norm_eps=1e-6,
            **kwargs,
        )

    @classmethod
    def olmo2_13B(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        """
        A 13B OLMo model config.
        """
        return cls.llama2_13B(
            vocab_size,
            block_name=kwargs.pop("block_name", TransformerBlockType.reordered_norm),
            qk_norm=kwargs.pop("qk_norm", True),
            rope_theta=kwargs.pop("rope_theta", 500_000),
            layer_norm_eps=1e-6,
            **kwargs,
        )

    @classmethod
    def olmo2_32B(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        """
        A 32B OLMo model config.
        """
        d_model = 5120
        return cls.llama_like(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=kwargs.pop("n_layers", 64),
            n_heads=kwargs.pop("n_heads", 40),
            n_kv_heads=kwargs.pop("n_kv_heads", 8),
            block_name=kwargs.pop("block_name", TransformerBlockType.reordered_norm),
            qk_norm=kwargs.pop("qk_norm", True),
            rope_theta=kwargs.pop("rope_theta", 500_000),
            hidden_size_multiple_of=kwargs.pop("hidden_size_multiple_of", 512),
            hidden_size_multiplier=kwargs.pop("hidden_size_multiplier", 27648 / (8 * d_model / 3)),
            layer_norm_eps=1e-6,
            **kwargs,
        )

    @classmethod
    def smallmoe(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        d_model = kwargs.pop("d_model", 768)
        return cls.llama_like(
            d_model=d_model,
            vocab_size=vocab_size,
            n_layers=kwargs.pop("n_layers", 12),
            n_heads=kwargs.pop("n_heads", 12),
            name=kwargs.pop("name", TransformerType.moe),
            block_name=kwargs.pop("block_name", TransformerBlockType.moe_reordered_norm),
            qk_norm=kwargs.pop("qk_norm", True),
            rope_theta=kwargs.pop("rope_theta", 500_000),
            layer_norm_eps=1e-6,
            feed_forward_moe=MoEConfig(
                name=MoEType.default,
                num_experts=32,
                hidden_size=int(0.5 * d_model),
                router=MoERouterConfig(top_k=4),
                shared_mlp=FeedForwardConfig(hidden_size=d_model * 2),
                lb_loss_weight=0.01,
                z_loss_weight=0.001,
            ),
        )

    @classmethod
    def small_hybrid_moe(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        d_model = kwargs.pop("d_model", 768)
        return cls.llama_like(
            d_model=d_model,
            vocab_size=vocab_size,
            n_layers=kwargs.pop("n_layers", 12),
            n_heads=kwargs.pop("n_heads", 12),
            name=kwargs.pop("name", TransformerType.moe),
            block_name=kwargs.pop("block_name", TransformerBlockType.moe_hybrid_reordered_norm),
            qk_norm=kwargs.pop("qk_norm", True),
            rope_theta=kwargs.pop("rope_theta", 500_000),
            layer_norm_eps=1e-6,
            feed_forward=FeedForwardConfig(hidden_size=d_model * 2, bias=False),
            feed_forward_moe=MoEConfig(
                name=MoEType.default,
                num_experts=32,
                hidden_size=int(0.5 * d_model),
                router=MoERouterConfig(top_k=4),
                lb_loss_weight=0.01,
                z_loss_weight=0.001,
            ),
        )

    @classmethod
    def olmoe_1B_7B(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        d_model = kwargs.pop("d_model", 2048)
        return cls.llama_like(
            d_model=d_model,
            vocab_size=vocab_size,
            n_layers=kwargs.pop("n_layers", 16),
            n_heads=kwargs.pop("n_heads", 16),
            name=kwargs.pop("name", TransformerType.moe),
            block_name=kwargs.pop("block_name", TransformerBlockType.moe_reordered_norm),
            qk_norm=kwargs.pop("qk_norm", True),
            rope_theta=kwargs.pop("rope_theta", 500_000),
            layer_norm_eps=1e-6,
            feed_forward_moe=MoEConfig(
                name=MoEType.dropless,
                num_experts=64,
                hidden_size=int(0.5 * d_model),
                router=MoERouterConfig(top_k=8),
                lb_loss_weight=0.01,
                z_loss_weight=0.001,
            ),
        )

    @classmethod
    def ngpt_271M(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        """
        A 271M nGPT model config.
        """
        return cls.ngpt_like(
            d_model=1024,
            vocab_size=vocab_size,
            n_layers=kwargs.pop("n_layers", 16),
            n_heads=kwargs.pop("n_heads", 16),
            **kwargs,
        )

    @classmethod
    def ngpt_1B(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        """
        A 1B nGPT model config.
        """
        return cls.ngpt_like(
            d_model=2048,
            vocab_size=vocab_size,
            n_layers=kwargs.pop("n_layers", 18),
            n_heads=kwargs.pop("n_heads", 16),
            **kwargs,
        )

    @classmethod
    def llama2_271M(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        """
        A 271M Llama2-like model config.
        """
        return cls.llama_like(
            d_model=1024,
            vocab_size=vocab_size,
            n_layers=kwargs.pop("n_layers", 16),
            n_heads=kwargs.pop("n_heads", 8),
            rope_theta=kwargs.pop("rope_theta", 10_000),
            **kwargs,
        )

    @classmethod
    def llama2_1B(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        """
        A 1B Llama2-like model config.

        Note: Llama2 doesn't have a 1B. We made this up.
        """
        return cls.llama_like(
            d_model=2048,
            vocab_size=vocab_size,
            n_layers=kwargs.pop("n_layers", 18),
            n_heads=kwargs.pop("n_heads", 16),
            rope_theta=kwargs.pop("rope_theta", 10_000),
            **kwargs,
        )

    @classmethod
    def llama2_7B(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        """
        A 7B Llama2-like model config.
        """
        return cls.llama_like(
            d_model=4096,
            vocab_size=vocab_size,
            n_layers=kwargs.pop("n_layers", 32),
            n_heads=kwargs.pop("n_heads", 32),
            rope_theta=kwargs.pop("rope_theta", 10_000),
            **kwargs,
        )

    @classmethod
    def llama2_13B(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        """
        A 7B Llama2-like model config.
        """
        return cls.llama_like(
            d_model=5120,
            vocab_size=vocab_size,
            n_layers=kwargs.pop("n_layers", 40),
            n_heads=kwargs.pop("n_heads", 40),
            rope_theta=kwargs.pop("rope_theta", 10_000),
            **kwargs,
        )

    @classmethod
    def llama2_26B(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        """
        A 26B Llama2-like model config.
        """
        return cls.llama_like(
            d_model=5120,
            vocab_size=vocab_size,
            n_layers=kwargs.pop("n_layers", 80),
            n_heads=kwargs.pop("n_heads", 40),
            rope_theta=kwargs.pop("rope_theta", 10_000),
            **kwargs,
        )

    @classmethod
    def llama2_70B(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        """
        A 70B Llama2-like model config.
        """
        return cls.llama_like(
            d_model=8192,
            vocab_size=vocab_size,
            n_layers=kwargs.pop("n_layers", 80),
            n_heads=kwargs.pop("n_heads", 64),
            n_kv_heads=kwargs.pop("n_kv_heads", 8),
            rope_theta=kwargs.pop("rope_theta", 10_000),
            hidden_size_multiplier=1.3,
            hidden_size_multiple_of=4096,
            **kwargs,
        )

    @classmethod
    def llama3_1B(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        """
        A 1B Llama3-like model config.
        """
        return cls.llama_like(
            d_model=2048,
            vocab_size=vocab_size,
            n_layers=kwargs.pop("n_layers", 16),
            n_heads=kwargs.pop("n_heads", 32),
            n_kv_heads=kwargs.pop("n_kv_heads", 8),
            rope_theta=kwargs.pop("rope_theta", 500_000),
            hidden_size_multiplier=1.5,
            **kwargs,
        )

    @classmethod
    def llama3_8B(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        """
        An 8B Llama3-like model config.
        """
        return cls.llama_like(
            d_model=4096,
            vocab_size=vocab_size,
            n_layers=kwargs.pop("n_layers", 32),
            n_heads=kwargs.pop("n_heads", 32),
            n_kv_heads=kwargs.pop("n_kv_heads", 8),
            rope_theta=kwargs.pop("rope_theta", 500_000),
            hidden_size_multiplier=1.3,
            hidden_size_multiple_of=1024,
            **kwargs,
        )

    @classmethod
    def llama3_70B(cls, vocab_size: int, **kwargs) -> "TransformerConfig":
        """
        A 70B Llama3-like model config.
        """
        return cls.llama_like(
            d_model=8196,
            vocab_size=vocab_size,
            n_layers=kwargs.pop("n_layers", 80),
            n_heads=kwargs.pop("n_heads", 64),
            n_kv_heads=kwargs.pop("n_kv_heads", 8),
            rope_theta=kwargs.pop("rope_theta", 500_000),
            hidden_size_multiplier=1.3,
            hidden_size_multiple_of=4096,
            **kwargs,
        )

    @classmethod
    def llama3_405B(
        cls,
        vocab_size: int,
        **kwargs,
    ) -> "TransformerConfig":
        """
        A 405B Llama3-like model config.
        """
        return cls.llama_like(
            d_model=16384,
            vocab_size=vocab_size,
            n_layers=kwargs.pop("n_layers", 126),
            n_heads=kwargs.pop("n_heads", 128),
            n_kv_heads=kwargs.pop("n_kv_heads", 8),
            rope_theta=kwargs.pop("rope_theta", 500_000),
            hidden_size_multiplier=1.2,
            hidden_size_multiple_of=4096,
            **kwargs,
        )

    @classmethod
    def llama_like(
        cls,
        *,
        d_model: int,
        vocab_size: int,
        n_layers: int,
        n_heads: int,
        n_kv_heads: Optional[int] = None,
        qk_norm: bool = False,
        layer_norm_eps: float = 1e-5,
        rope_theta: int = 500_000,
        rope_type: Optional[RoPEType] = None,
        hidden_size_multiple_of: int = 256,
        hidden_size_multiplier: Optional[float] = None,
        fused_ops: bool = False,
        use_flash: bool = False,
        block_name: TransformerBlockType = TransformerBlockType.default,
        block_mods: Optional[
            Dict[int, Callable[[TransformerBlockConfig], TransformerBlockConfig]]
        ] = None,
        dtype: DType = DType.float32,
        rope_scaling: Optional[RoPEScalingConfig] = None,
        feed_forward: Optional[FeedForwardConfig] = None,
        feed_forward_moe: Optional[MoEConfig] = None,
        **kwargs,
    ) -> "TransformerConfig":
        """
        Create a Llama-like model configuration.

        :param hidden_size_multiple_of: Ensure the FFN hidden size is a multiple of this value.
        :param hidden_size_multiplier: Custom multiplier for the FFN hidden size.
        :param fused_ops: Use fused operations where possible.
        :param use_flash: Use flash-attn.
        :param block_mods: A dictionary of block indices to functions that take the base block config and return a modified block config.
        :param dtype: The default data type to use for all parameters.
        """
        # Resolve hidden size of FFN in blocks.
        hidden_size = int(8 * d_model / 3)
        if hidden_size_multiplier is not None:
            hidden_size = int(hidden_size_multiplier * hidden_size)
        hidden_size = ensure_multiple_of(hidden_size, hidden_size_multiple_of)

        # Configure global layer norm.
        layer_norm = LayerNormConfig(
            name=LayerNormType.fused_rms if fused_ops else LayerNormType.rms,
            eps=layer_norm_eps,
            bias=False,
            dtype=dtype,
        )

        # Decide on attention/rope implementations.
        att_type = AttentionType.default
        if rope_type is None:
            rope_type = RoPEType.default
            if fused_ops and n_kv_heads is None:  # fused attention not compatible with MQA/GQA.
                att_type = AttentionType.fused
                rope_type = RoPEType.fused

        # Feed-forward.
        if feed_forward is None and feed_forward_moe is None:
            feed_forward = FeedForwardConfig(hidden_size=hidden_size, bias=False, dtype=dtype)

        # Configure blocks.
        block = TransformerBlockConfig(
            name=block_name,
            attention=AttentionConfig(
                name=att_type,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                bias=False,
                rope=RoPEConfig(name=rope_type, theta=rope_theta, scaling=rope_scaling),
                qk_norm=layer_norm if qk_norm else None,
                use_flash=use_flash,
                dtype=dtype,
            ),
            feed_forward=feed_forward,
            feed_forward_moe=feed_forward_moe,
            layer_norm=layer_norm,
        )

        if block_mods and kwargs.get("block_overrides"):
            raise OLMoConfigurationError(
                "`block_mods` and `block_overrides` cannot be used together."
            )
        block_overrides = None
        if block_mods:
            block_overrides = {i: block_mods[i](block.copy()) for i in block_mods}
        elif kwargs.get("block_overrides"):
            block_overrides = kwargs.get("block_overrides")

        return cls(
            d_model=d_model,
            vocab_size=vocab_size,
            n_layers=n_layers,
            block=block,
            lm_head=LMHeadConfig(layer_norm=layer_norm, bias=False, dtype=dtype),
            dtype=dtype,
            block_overrides=block_overrides,
            **kwargs,
        )

    @classmethod
    def llama_like_moe(
        cls,
        *,
        d_model: int,
        vocab_size: int,
        n_layers: int,
        n_heads: int,
        num_experts: int,
        top_k: int,
        expert_hidden_size: int,
        shared_expert_hidden_size: Optional[int] = None,
        dropless: bool = False,
        capacity_factor: Optional[float] = None,
        lb_loss_weight: float = 0.01,
        z_loss_weight: Optional[float] = 0.001,
        reordered_norm: bool = False,
        hybrid: bool = False,
        **kwargs,
    ) -> "TransformerConfig":
        block_name: TransformerBlockType
        if reordered_norm:
            block_name = (
                TransformerBlockType.moe_hybrid_reordered_norm
                if hybrid
                else TransformerBlockType.moe_reordered_norm
            )
        else:
            block_name = TransformerBlockType.moe_hybrid if hybrid else TransformerBlockType.moe
        return cls.llama_like(
            d_model=d_model,
            vocab_size=vocab_size,
            n_layers=n_layers,
            n_heads=n_heads,
            name=TransformerType.moe,
            block_name=block_name,
            qk_norm=kwargs.pop("qk_norm", reordered_norm),
            feed_forward_moe=MoEConfig(
                name=MoEType.default if not dropless else MoEType.dropless,
                num_experts=num_experts,
                hidden_size=expert_hidden_size,
                capacity_factor=capacity_factor,
                router=MoERouterConfig(top_k=top_k),
                shared_mlp=None
                if shared_expert_hidden_size is None
                else FeedForwardConfig(hidden_size=shared_expert_hidden_size, bias=False),
                lb_loss_weight=lb_loss_weight,
                z_loss_weight=z_loss_weight,
            ),
            **kwargs,
        )

    @classmethod
    def ngpt_like(
        cls,
        *,
        d_model: int,
        vocab_size: int,
        n_layers: int,
        n_heads: int,
        n_kv_heads: Optional[int] = None,
        qk_norm: bool = True,
        rope_theta: int = 500_000,
        hidden_size_multiple_of: int = 256,
        hidden_size_multiplier: Optional[float] = None,
        use_flash: bool = False,
        dtype: DType = DType.float32,
        **kwargs,
    ) -> "TransformerConfig":
        """
        Create an nGPT-like model configuration.
        """
        # Resolve hidden size of FFN in blocks.
        hidden_size = int(8 * d_model / 3)
        if hidden_size_multiplier is not None:
            hidden_size = int(hidden_size_multiplier * hidden_size)
        hidden_size = ensure_multiple_of(hidden_size, hidden_size_multiple_of)

        # Configure blocks.
        block = TransformerBlockConfig(
            name=TransformerBlockType.normalized,
            attention=AttentionConfig(
                name=AttentionType.normalized,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                qk_norm=None if not qk_norm else LayerNormConfig(name=LayerNormType.l2_norm),
                rope=RoPEConfig(name=RoPEType.default, theta=rope_theta),
                use_flash=use_flash,
                dtype=dtype,
            ),
            feed_forward=FeedForwardConfig(
                name=FeedForwardType.normalized, hidden_size=hidden_size, dtype=dtype
            ),
        )

        return cls(
            name=TransformerType.normalized,
            d_model=d_model,
            vocab_size=vocab_size,
            n_layers=n_layers,
            block=block,
            lm_head=LMHeadConfig(name=LMHeadType.normalized, dtype=dtype),
            dtype=dtype,
            init_method=InitMethod.normalized,
            **kwargs,
        )
