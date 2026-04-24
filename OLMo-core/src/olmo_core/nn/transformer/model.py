import json
import logging
import math
from collections import defaultdict
from itertools import chain
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    List,
    Literal,
    NamedTuple,
    Optional,
    Set,
    Tuple,
    Union,
    cast,
)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributed import DeviceMesh
from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, parallelize_module

from olmo_core.data.utils import get_cumulative_document_lengths
from olmo_core.distributed.parallel import get_pp_mesh
from olmo_core.distributed.utils import distribute_like, get_local_tensor, hide_from_torch, unhide_from_torch
from olmo_core.doc_utils import beta_feature
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.float8 import Float8Config
from olmo_core.utils import get_default_device, mark_dynamic, move_to_device

from ..attention import (
    Attention,
    FusedAttention,
    RingAttentionLoadBalancer,
    RingAttentionLoadBalancerType,
    ShortConvParams,
    compute_shortconv_delta,
)
from ..buffer_cache import BufferCache
from ..embedding_injection.engram import (
    EngramModule,
    apply_engram_pre_block,
    build_engram_modules,
)
from ..embedding_injection.ops.hash_injection import HashTokenMapInjection
from ..embedding_injection.metrics import (
    _warmup_scale_to_python_float,
)
from ..embedding_injection.mort import build_mort_modules, init_mort_modules, prepare_mort_block_kwargs
from ..embedding_injection.ops.shortconv import SwiGLUShortConv
from ..embedding_injection.retoken import build_retoken_modules, init_retoken_modules, prepare_retoken_block_kwargs
from ..embedding_injection.runtime import InjectionBlockContext
from ..embedding_injection.xgram import build_xgram_modules, prepare_xgram_block_kwargs
from ..functional import l2_normalize
from ..lm_head import LMHeadConfig, LMOutputWithLoss
from ..moe import MoEBase
from ..rope import RoPEBuffers, RotaryEmbeddingBase
from ..utils import selective_checkpointing_context_fn
from .block import (
    MoETransformerBlock,
    NormalizedTransformerBlock,
    TransformerBlock,
    TransformerBlockBase,
)
from .config import (
    TransformerActivationCheckpointingMode,
    TransformerBlockConfig,
    TransformerDataParallelWrappingStrategy,
    TransformerEmbeddingInjectionConfig,
)
from .init import InitMethod

if TYPE_CHECKING:
    from olmo_core.train.common import ReduceType

__all__ = [
    "Transformer",
    "NormalizedTransformer",
    "MoETransformer",
    "TransformerDataParallelWrappingStrategy",
    "TransformerActivationCheckpointingMode",
]


log = logging.getLogger(__name__)

_XGRAM_TARGET_SET = {"h", "q", "k", "v", "o"}
_QKVO_TARGET_SET = {"q", "k", "v", "o"}


def _embedding_like_numel(module: Optional[nn.Module]) -> int:
    if module is None:
        return 0
    if isinstance(module, nn.Embedding):
        return int(module.weight.numel())
    if isinstance(module, HashTokenMapInjection):
        total = int(module._bucket_embedding.weight.numel())
        total += sum(int(emb.weight.numel()) for emb in module._scalar_weight_embeddings)
        return total
    if isinstance(module, EngramModule):
        return sum(int(emb.weight.numel()) for emb in module.ngram_embeddings)
    return 0


def _parse_ordered_injection_targets(
    raw: Optional[str],
    *,
    valid: Set[str],
    default: Optional[List[str]] = None,
    env_name: str,
) -> List[str]:
    if raw is None or not raw.strip():
        return list(default) if default is not None else []

    tokens = [token.strip().lower() for token in raw.split(",") if token.strip()]
    targets: List[str] = []
    invalid: List[str] = []
    for token in tokens:
        if token in valid:
            if token not in targets:
                targets.append(token)
        else:
            invalid.append(token)
    if invalid:
        log.warning(
            "Invalid %s entries: %s (valid: %s)",
            env_name,
            ",".join(invalid),
            ",".join(sorted(valid)),
        )
    if targets:
        return targets
    return list(default) if default is not None else []


def _resolve_injection_target_state(
    *,
    injection_version: str,
    injection_targets_raw: Optional[str],
) -> Tuple[List[str], List[str], bool]:
    if injection_version == "X-gram":
        injection_targets = _parse_ordered_injection_targets(
            injection_targets_raw,
            valid=_XGRAM_TARGET_SET,
            default=["h"],
            env_name="INJECTION_TARGETS",
        )
        qkvo_targets = [target for target in injection_targets if target in _QKVO_TARGET_SET]
        return injection_targets, qkvo_targets, "h" in injection_targets

    return [], [], False


class InjectionDepthScaleSpec(NamedTuple):
    buffer_name: str
    scale: float


class Transformer(nn.Module):
    """
    A typical "Llama-style" transformer implementation.

    :param d_model: The model dimensionality.
    :param vocab_size: The vocab size.
    :param n_layers: The number of transformer layers/blocks.
    :param block: The block configuration.
    :param layer_norm: The layer norm config for the final layer norm.
    :param bias: Whether to use a bias in the final linear layer.
    :param dtype: The datatype to use for the linear output layer.
    :param init_device: The device used when initializing parameters.
    """

    def __init__(
        self,
        *,
        d_model: int,
        vocab_size: int,
        n_layers: int,
        block: TransformerBlockConfig,
        lm_head: LMHeadConfig,
        dtype: torch.dtype = torch.float32,
        init_method: InitMethod = InitMethod.normal,
        init_device: str = "cpu",
        init_seed: int = 0,
        init_std: float = 0.02,
        block_overrides: Optional[Dict[int, TransformerBlockConfig]] = None,
        embedding_injection: Optional[TransformerEmbeddingInjectionConfig] = None,
        mask_indices_path: Optional[Path] = None,
        mort_aux_loss_weight: float = 1e-4,
    ):
        super().__init__()

        cache = BufferCache()

        self.d_model = d_model
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.n_attn_heads = block.attention.n_heads
        self.dtype = dtype

        self.embeddings = nn.Embedding(vocab_size, d_model, dtype=dtype, device=init_device)
        self.blocks = nn.ModuleDict()
        for block_idx in range(n_layers):
            block_config = block
            if block_overrides is not None and block_idx in block_overrides:
                block_config = block_overrides[block_idx]
            self.blocks[str(block_idx)] = self._validate_block(
                block_config.build(
                    d_model=d_model,
                    block_idx=block_idx,
                    n_layers=n_layers,
                    init_device=init_device,
                    cache=cache,
                )
            )
        self.lm_head = lm_head.build(
            d_model=d_model, vocab_size=vocab_size, init_device=init_device
        )

        self.init_device = init_device
        self.init_method = InitMethod(init_method)
        self.init_seed = init_seed
        self.init_std = init_std
        self._embedding_injection_config = embedding_injection
        self._injection_h_embeddings = nn.ModuleDict()
        self._injection_h_gates = nn.ModuleDict()
        self._injection_h_gate_defaults: Dict[str, List[float]] = defaultdict(list)
        self._injection_version = "None"
        self._retoken_embeddings = nn.ModuleDict()
        self._retoken_scalers = nn.ModuleDict()
        self._injection_h_shortconvs = nn.ModuleDict()
        self._engram_enabled = False
        self._engram_config: Optional[Dict[str, Any]] = None
        self._mort_aux_loss_weight = mort_aux_loss_weight
        self._injection_qk_embeddings = nn.ModuleDict()
        self._injection_qk_gates = nn.ModuleDict()
        self._injection_qk_shortconvs = nn.ModuleDict()
        self._injection_q_embeddings = nn.ModuleDict()
        self._injection_q_gates = nn.ModuleDict()
        self._injection_q_shortconvs = nn.ModuleDict()
        self._injection_k_embeddings = nn.ModuleDict()
        self._injection_k_gates = nn.ModuleDict()
        self._injection_k_shortconvs = nn.ModuleDict()
        self._injection_v_embeddings = nn.ModuleDict()
        self._injection_v_gates = nn.ModuleDict()
        self._injection_v_shortconvs = nn.ModuleDict()
        self._injection_o_embeddings = nn.ModuleDict()
        self._injection_o_gates = nn.ModuleDict()
        self._injection_o_shortconvs = nn.ModuleDict()
        injection_version = (
            getattr(embedding_injection, "mode", None)
            if embedding_injection is not None
            else None
        ) or "None"
        injection_targets_cfg = (
            list(getattr(embedding_injection, "targets", None) or [])
            if embedding_injection is not None
            else []
        )
        injection_targets_raw: Optional[str] = (
            ",".join(injection_targets_cfg) if injection_targets_cfg else None
        )
        self._hash_token_map_path = (
            getattr(embedding_injection, "hash_token_map_path", None)
            if embedding_injection is not None
            else None
        )
        self._injection_log_interval = max(
            0,
            int(getattr(embedding_injection, "log_interval", 100))
            if embedding_injection is not None
            else 100,
        )
        self._log_injection_wandb_missing = False
        self._log_injection_wandb_inactive = False
        self._log_baseline_wandb_missing = False
        self._log_baseline_wandb_inactive = False
        self._injection_depth_scale_disabled = bool(
            getattr(embedding_injection, "depth_scale_disabled", False)
            if embedding_injection is not None
            else False
        )
        self._hash_injection_enabled = bool(
            getattr(embedding_injection, "hash_enabled", False)
            if embedding_injection is not None
            else False
        )
        self._injection_depth_scale_specs: Dict[str, List[InjectionDepthScaleSpec]] = {}
        self._injection_targets: List[str] = []
        self._h_target_enabled = False

        # Default two-head multi-table hash multipliers. Adjacent entries are
        # selected in a round-robin pattern by table index.
        default_hash_head_multipliers_k2 = [
            11400714819323198485,
            14313749767032793493,
            18397679294719823053,
            13627969195753728349,
            7109453100751438417,
            1609587929392839161,
            15404834773449339307,
            10041221585803275913,
        ]
        default_hash_num_heads = 2

        # Shortconv kernel schedule shared by H/QK/Q/K/V/O injection paths.
        kernels = (
            list(getattr(embedding_injection, "shortconv_kernels", None) or [])
            if embedding_injection is not None
            else []
        )
        self._sc_multi_scale_kernels: List[int] = kernels if kernels else [3, 5, 7, 9]

        def _shortconv_kernel_for_idx(
            idx: int,
        ) -> int:
            kernels_list = self._sc_multi_scale_kernels
            chosen = kernels_list[idx] if idx < len(kernels_list) else kernels_list[idx % len(kernels_list)]
            return max(1, int(chosen))

        self._shortconv_enabled = bool(
            getattr(embedding_injection, "shortconv_enabled", False)
            if embedding_injection is not None
            else False
        )
        # Injection shortconv always uses RMSNorm + SwiGLU.
        self._sc_swiglu_gate_bias = 1.0
        self._shortconv_rmsnorm_eps = 1e-5
        # Track last logged step per layer for injection metrics to avoid duplicate logs within a step.
        self._injection_logged_steps: Dict[int, int] = {}
        # Track last logged step per layer for baseline (no-injection) metrics.
        self._baseline_logged_steps: Dict[int, int] = {}

        self._injection_version = injection_version
        self._injection_lambda_init = float(
            getattr(embedding_injection, "lambda_init", 1.0) or 1.0
            if embedding_injection is not None
            else 1.0
        )
        self._injection_lambda_warmup_enabled = bool(
            getattr(embedding_injection, "lambda_warmup_enabled", False)
            if embedding_injection is not None
            else False
        )
        self._injection_lambda_warmup_steps = max(
            0,
            int(getattr(embedding_injection, "lambda_warmup_steps", 0) or 0)
            if embedding_injection is not None
            else 0,
        )
        if (
            injection_version == "X-gram"
            and self._injection_lambda_warmup_enabled
            and self._injection_lambda_warmup_steps == 0
        ):
            raise OLMoConfigurationError(
                "INJECTION_LAMBDA_WARMUP_ENABLE=1 but no valid warmup steps inferred; "
                "please set a positive lambda_warmup_steps in TransformerEmbeddingInjectionConfig."
            )

        # Engram: parse configuration from TransformerEmbeddingInjectionConfig.
        self._engram_enabled = injection_version == "Engram"
        if self._engram_enabled:
            _engram_tokenizer_id = (
                getattr(embedding_injection, "engram_tokenizer_id", None)
                if embedding_injection is not None
                else None
            )
            if not _engram_tokenizer_id:
                raise OLMoConfigurationError(
                    "Engram mode requires engram_tokenizer_id in TransformerEmbeddingInjectionConfig"
                )
            _engram_cache_path = getattr(embedding_injection, "engram_cache_path", None)
            _engram_base_seed = int(getattr(embedding_injection, "engram_base_seed", 42))
            _engram_hc_mult = int(getattr(embedding_injection, "engram_hc_mult", 1))
            _engram_shortconv_enabled = bool(
                getattr(embedding_injection, "engram_shortconv_enabled", True)
            )
            _engram_shortconv_kernel = int(
                getattr(embedding_injection, "engram_shortconv_kernel", 4)
            )
            _engram_shortconv_dilation = int(
                getattr(embedding_injection, "engram_shortconv_dilation", 1)
            )
            _engram_shortconv_activation = bool(
                getattr(embedding_injection, "engram_shortconv_activation", True)
            )

            # Engram mode: set-based parsing for arbitrary n-gram combinations
            # Examples: "2gram", "2gram+3gram", "2gram+4gram"
            _engram_mode = getattr(embedding_injection, "engram_mode", "2gram+3gram")
            _engram_mode_parts = {s.strip() for s in _engram_mode.split("+") if s.strip()}
            _engram_ngram_levels: List[int] = sorted(
                int(p.replace("gram", "")) for p in _engram_mode_parts
            )
            if len(_engram_ngram_levels) == 0:
                raise OLMoConfigurationError("Engram mode requires at least one n-gram level")
            if any(level < 2 for level in _engram_ngram_levels):
                raise OLMoConfigurationError("Engram mode only supports n-gram levels >= 2")
            _num_active_ngrams = len(_engram_ngram_levels)
            _engram_dim_per_ngram_cfg = getattr(embedding_injection, "engram_dim_per_ngram", None)
            _engram_dim_per_ngram = (
                int(_engram_dim_per_ngram_cfg)
                if _engram_dim_per_ngram_cfg is not None
                else d_model // _num_active_ngrams
            )
            # N-gram config (used for all n-gram levels >= 2)
            _engram_ngram_heads = int(getattr(embedding_injection, "engram_ngram_heads", 4))
            _engram_ngram_target = getattr(
                embedding_injection,
                "engram_ngram_target_buckets",
                75968,
            )
            _engram_ngram_seed = int(getattr(embedding_injection, "engram_ngram_seed", 137))
            self._engram_config = {
                "tokenizer_id": _engram_tokenizer_id,
                "cache_path": _engram_cache_path,
                "base_seed": _engram_base_seed,
                "hc_mult": _engram_hc_mult,
                "shortconv_enabled": _engram_shortconv_enabled,
                "shortconv_kernel": _engram_shortconv_kernel,
                "shortconv_dilation": _engram_shortconv_dilation,
                "shortconv_activation": _engram_shortconv_activation,
                "ngram_levels": _engram_ngram_levels,
                "ngram_heads": _engram_ngram_heads,
                "ngram_target_capacity": _engram_ngram_target,
                "ngram_dim": _engram_dim_per_ngram,
                "ngram_seed": _engram_ngram_seed,
            }
            log.info(
                "Engram config prepared (mode=%s dim_per_ngram=%d hc_mult=%d "
                "shortconv=%s kernel=%d tokenizer=%s "
                "ngram_levels=%s ngram_heads=%d ngram_target=%s)",
                _engram_mode,
                _engram_dim_per_ngram,
                _engram_hc_mult,
                _engram_shortconv_enabled,
                _engram_shortconv_kernel,
                _engram_tokenizer_id,
                _engram_ngram_levels,
                _engram_ngram_heads,
                _engram_ngram_target,
            )

        self._injection_targets, self._attention_injection_targets, self._h_target_enabled = _resolve_injection_target_state(
            injection_version=injection_version,
            injection_targets_raw=injection_targets_raw,
        )
        attention_has_qkv = any(t in {"q", "k", "v"} for t in self._attention_injection_targets)
        attention_qkv_targets = [t for t in self._attention_injection_targets if t in {"q", "k", "v"}]
        attention_qkv_target_set = set(attention_qkv_targets)
        self._attention_qk_sharing = bool(
            getattr(embedding_injection, "qk_sharing", False)
            if embedding_injection is not None
            else False
        )
        attention_qk_sharing_active = self._attention_qk_sharing and {"q", "k"}.issubset(attention_qkv_target_set)
        if self._attention_qk_sharing:
            if attention_qk_sharing_active:
                log.info("Attention QK sharing enabled: q and k will share the same injection embedding/gate/shortconv")
                for name in (
                    "_injection_q_embeddings",
                    "_injection_q_gates",
                    "_injection_q_shortconvs",
                    "_injection_k_embeddings",
                    "_injection_k_gates",
                    "_injection_k_shortconvs",
                ):
                    self._modules.pop(name, None)
                    setattr(self, name, None)
            else:
                log.info("Attention QK sharing requested but inactive because both q and k are not present in INJECTION_TARGETS")

        if embedding_injection is not None:
            if injection_version == "X-gram":
                build_xgram_modules(
                    self,
                    embedding_injection,
                    vocab_size=vocab_size,
                    d_model=d_model,
                    n_layers=n_layers,
                    dtype=dtype,
                    init_device=init_device,
                    init_std=self.init_std,
                    hash_num_heads=default_hash_num_heads,
                    hash_multipliers=default_hash_head_multipliers_k2,
                )
            elif injection_version == "Engram":
                build_engram_modules(
                    self,
                    embedding_injection,
                    vocab_size=vocab_size,
                    d_model=d_model,
                    dtype=dtype,
                    init_device=init_device,
                )
            elif injection_version == "Retoken":
                build_retoken_modules(
                    self,
                    embedding_injection,
                    vocab_size=vocab_size,
                    d_model=d_model,
                    dtype=dtype,
                    init_device=init_device,
                )
            elif injection_version == "Mort":
                build_mort_modules(
                    self,
                    embedding_injection,
                    vocab_size=vocab_size,
                    d_model=d_model,
                    dtype=dtype,
                    init_device=init_device,
                    mort_aux_loss_weight=mort_aux_loss_weight,
                )
        else:
            self._injection_h_embeddings = nn.ModuleDict()
            self._injection_h_gates = nn.ModuleDict()
            self._retoken_embeddings = nn.ModuleDict()
            self._retoken_scalers = nn.ModuleDict()

        self._mask_indices_path = None

        if getattr(self, "_mask_indices_path", None) is not None:
            self.mask_low_frequency_injections(self._mask_indices_path)

        self._cache = cache
        self._pp_enabled = False
        self._pp_group_size = 1
        self._fp8_enabled = False
        self._precompute_float8_dynamic_scale_for_fsdp = False
        self._compile_enabled = False
        self._device: Optional[torch.device] = None
        self._cp_load_balancer: Optional[RingAttentionLoadBalancer] = None
        self._tp_enabled = False
        self._tp_mesh: Optional[DeviceMesh] = None
        self._fsdp_enabled = False
        self._cached_num_params: Optional[int] = None
        self._cached_num_non_embedding_params: Optional[int] = None

        if injection_version == "Mort":
            self._log_mort_sparse_modules()

    def _log_mort_sparse_modules(self) -> None:
        """
        Convenience helper to print shapes of mort_sparse weight_generators/scalers.
        """
        for block_idx, block in self.blocks.items():
            block = cast(TransformerBlock, block)
            if block.mort_sparse_weight_generator is None and block.mort_sparse_scaler is None:
                continue
            wg_shape = (
                tuple(block.mort_sparse_weight_generator.weight.shape)  # type: ignore[arg-type]
                if block.mort_sparse_weight_generator is not None
                else None
            )
            sc_shape = (
                tuple(block.mort_sparse_scaler.shape)  # type: ignore[arg-type]
                if block.mort_sparse_scaler is not None
                else None
            )
            log.info(
                "Mort block %s: weight_generator=%s scaler=%s top_k=%s",
                block_idx,
                wg_shape,
                sc_shape,
                block.mort_sparse_top_k,
            )

    def _qkvo_embedding_module_dicts(self) -> Tuple[nn.ModuleDict, ...]:
        return tuple(
            module_dict
            for module_dict in (
                self._injection_qk_embeddings,
                self._injection_q_embeddings,
                self._injection_k_embeddings,
                self._injection_v_embeddings,
                self._injection_o_embeddings,
            )
            if module_dict is not None
        )

    def _qkvo_shortconv_module_dicts(self) -> Tuple[nn.ModuleDict, ...]:
        return tuple(
            module_dict
            for module_dict in (
                self._injection_qk_shortconvs,
                self._injection_q_shortconvs,
                self._injection_k_shortconvs,
                self._injection_v_shortconvs,
                self._injection_o_shortconvs,
            )
            if module_dict is not None
        )

    def _all_injection_embedding_module_dicts(
        self,
        *,
        include_retoken: bool = False,
    ) -> Tuple[nn.ModuleDict, ...]:
        dicts: List[nn.ModuleDict] = [self._injection_h_embeddings, *self._qkvo_embedding_module_dicts()]
        if include_retoken:
            dicts.append(self._retoken_embeddings)
        return tuple(dicts)

    def mask_low_frequency_injections(self, mask_indices_path: Path) -> None:
        mask_tensor = self._resolve_mask_indices(mask_indices_path)
        if mask_tensor.numel() == 0:
            return

        total_masked = 0
        for embeddings in chain(
            self._injection_h_embeddings.values(),
            self._retoken_embeddings.values(),
            *(module_dict.values() for module_dict in self._qkvo_embedding_module_dicts()),
        ):
            for embedding in embeddings:
                total_masked += self._mask_embedding_rows(embedding, mask_tensor)
        log.info("Injected %d parameters now excluded from gradients via low-frequency mask.", total_masked)

    def _resolve_mask_indices(self, mask_indices_path: Path) -> torch.Tensor:
        data = np.load(mask_indices_path)
        arr_key = "low_indices"
        if arr_key not in data.files:
            arr_key = data.files[0]
        arr = np.asarray(data[arr_key])
        tensor = torch.from_numpy(arr).long()
        return torch.unique(tensor)

    def _mask_embedding_rows(self, embedding: nn.Embedding, indices: torch.Tensor) -> int:
        device = embedding.weight.device
        indices = indices.to(device)
        with torch.no_grad():
            embedding.weight.index_fill_(0, indices, 0.0)

        if getattr(embedding.weight, "_masked_hook", False):
            return indices.numel() * embedding.weight.size(1)

        def _hook(grad: torch.Tensor) -> torch.Tensor:
            grad.index_fill_(0, indices, 0.0)
            return grad

        embedding.weight.register_hook(_hook)
        embedding.weight._masked_hook = True
        return indices.numel() * embedding.weight.size(1)

    def _reset_hash_injection_buffers(self, *, device: Optional[torch.device] = None) -> None:
        has_hash_injection = any(
            hasattr(module, "_reset_injection_buffers")
            for modules in self._all_injection_embedding_module_dicts()
            for modules in modules.values()
            for module in modules
        )
        if not has_hash_injection:
            return
        target_device = device or get_default_device()
        for module_dict in self._all_injection_embedding_module_dicts():
            for modules in module_dict.values():
                for module in modules:
                    if hasattr(module, "_reset_injection_buffers"):
                        module._reset_injection_buffers(device=target_device)

    def _register_injection_depth_scale(
        self,
        *,
        block_key: str,
        gate_idx: int,
        layer_idx: Optional[int],
        device: Union[str, torch.device],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        scale_val = math.sqrt((0 if layer_idx is None else max(layer_idx, 0)) + 1)
        buffer_name = f"_injection_depth_scale_{block_key}_{gate_idx}"
        scale_tensor = getattr(self, buffer_name, None)
        target_device: Union[str, torch.device] = device if str(device) != "meta" else "cpu"
        if scale_tensor is None:
            scale_tensor = torch.tensor(
                [scale_val],
                dtype=dtype,
                device=target_device,
            )
            self.register_buffer(buffer_name, scale_tensor, persistent=False)
        else:
            if scale_tensor.is_meta:
                scale_tensor = torch.tensor(
                    [scale_val],
                    dtype=dtype,
                    device=target_device,
                )
                setattr(self, buffer_name, scale_tensor)
            else:
                if scale_tensor.device != torch.device(target_device):
                    scale_tensor = scale_tensor.to(device=target_device)
                    setattr(self, buffer_name, scale_tensor)
                if scale_tensor.dtype != dtype:
                    scale_tensor = scale_tensor.to(dtype=dtype)
                    setattr(self, buffer_name, scale_tensor)
                scale_tensor.fill_(scale_val)
        specs = self._injection_depth_scale_specs.setdefault(block_key, [])
        if not any(spec.buffer_name == buffer_name for spec in specs):
            specs.append(InjectionDepthScaleSpec(buffer_name=buffer_name, scale=scale_val))
        return scale_tensor

    def _get_injection_depth_scale(
        self,
        *,
        block_key: str,
        gate_idx: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        buffer_name = f"_injection_depth_scale_{block_key}_{gate_idx}"
        scale_tensor = getattr(self, buffer_name, None)
        if scale_tensor is None:
            return None
        if isinstance(scale_tensor, DTensor):
            scale_tensor = scale_tensor.to_local()
        if scale_tensor.device != device or scale_tensor.dtype != dtype:
            scale_tensor = scale_tensor.to(device=device, dtype=dtype)
            setattr(self, buffer_name, scale_tensor)
        return scale_tensor

    def _reset_injection_depth_scales(self, *, device: Optional[torch.device] = None) -> None:
        if not self._injection_depth_scale_specs:
            return
        target_device = device or get_default_device()
        for block_key, specs in self._injection_depth_scale_specs.items():
            for spec in specs:
                scale_tensor = getattr(self, spec.buffer_name, None)
                if scale_tensor is None or scale_tensor.is_meta:
                    scale_tensor = torch.tensor(
                        [spec.scale],
                        dtype=torch.float32,
                        device=target_device,
                    )
                    setattr(self, spec.buffer_name, scale_tensor)
                    if spec.buffer_name not in self._buffers:
                        self.register_buffer(spec.buffer_name, scale_tensor, persistent=False)
                else:
                    if scale_tensor.device != target_device:
                        scale_tensor = scale_tensor.to(device=target_device)
                        setattr(self, spec.buffer_name, scale_tensor)
                    scale_tensor.fill_(spec.scale)

    def _compute_shortconv_delta(
        self,
        *,
        src: torch.Tensor,
        conv: nn.Module,
        already_normalized: bool = False,
    ) -> torch.Tensor:
        """
        Apply the configured RMSNorm/activation around a depthwise shortconv and return only the delta.
        """
        return compute_shortconv_delta(
            src=src,
            conv=conv,
            params=ShortConvParams(
                rmsnorm_eps=self._shortconv_rmsnorm_eps,
            ),
            already_normalized=already_normalized,
        )

    def _apply_shortconv_residual(self, base: torch.Tensor, conv: nn.Module) -> torch.Tensor:
        """
        Apply optional RMSNorm and SiLU around a depthwise shortconv and add it back to the base tensor.
        """
        return base + self._compute_shortconv_delta(
            src=base,
            conv=conv,
            already_normalized=False,
        )

    def _normalize_injection(self, injection: torch.Tensor) -> torch.Tensor:
        """L2 normalize an injection embedding with numerical stability."""
        dtype = injection.dtype
        eps = torch.finfo(dtype).eps
        norm = torch.linalg.vector_norm(injection, dim=-1, keepdim=True)
        return injection / (norm + eps)

    def _validate_block(self, block: TransformerBlockBase) -> TransformerBlockBase:
        return block

    def compute_auxiliary_metrics(
        self, reset: bool = True
    ) -> Dict[str, Tuple[torch.Tensor, Optional["ReduceType"]]]:
        del reset
        return {}

    def reset_auxiliary_metrics(self):
        pass

    @property
    def pp_enabled(self) -> bool:
        return self._pp_enabled

    @property
    def fp8_enabled(self) -> bool:
        return self._fp8_enabled

    @property
    def tp_enabled(self) -> bool:
        return self._tp_enabled

    @property
    def fsdp_enabled(self) -> bool:
        return self._fsdp_enabled

    @property
    def is_moe(self) -> bool:
        return False

    @property
    def device(self) -> torch.device:
        if self._device is None:
            for p in self.parameters():
                if p.numel() > 0:
                    self._device = p.device
                    break
            else:
                self._device = get_default_device()
        return self._device

    @property
    def compile_enabled(self) -> bool:
        return self._compile_enabled

    def get_rope_buffers(
        self, seq_len: int, device: Optional[torch.device] = None
    ) -> Dict[int, Optional[RoPEBuffers]]:
        """
        Get the RoPE buffers to pass to each layer.
        """
        if device is None:
            device = self.device
        rope_buffers = {}
        for key, block in self.blocks.items():
            rope = cast(Optional[RotaryEmbeddingBase], block.attention.rope)  # type: ignore
            rope_buffers[int(key)] = None if rope is None else rope.get_buffers(seq_len, device)
        return rope_buffers

    @torch.no_grad()
    def init_weights(
        self,
        *,
        max_seq_len: Optional[int] = None,
        max_local_microbatch_size: Optional[int] = None,
        device: Optional[torch.device] = None,
        world_mesh: Optional[DeviceMesh] = None,
    ) -> torch.Generator:
        """
        Initialize the model weights.

        :param max_seq_len: The maximum sequence length expected. This is used
            to warm up the RoPE cache.
        :param max_local_microbatch_size: The maximum local (rank) micro-batch size (in tokens)
            expected. This is used to warm-up some MoE cache.
        :param device: The device the local copy of the model will be trained on.
        """
        device = device or self.device
        self.to_empty(device=device)

        self._reset_hash_injection_buffers(device=device)
        self._reset_injection_depth_scales(device=device)

        # Engram: reset lookup tables and hash buffers
        if self._engram_enabled:
            for modules in self._injection_h_embeddings.values():
                for module in modules:
                    if isinstance(module, EngramModule):
                        module.reset_buffers(device=device)

        for module in self.modules():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()  # type: ignore

        seed = self.init_seed
        if world_mesh is not None and self.pp_enabled:
            seed += get_pp_mesh(world_mesh).get_local_rank()

        generator = torch.Generator(device).manual_seed(seed)

        if self.embeddings is not None:
            self.init_method.init_embeddings(
                self.embeddings,
                d_model=self.d_model,
                std=self.init_std,
                generator=generator,
            )

            seen_init: Set[int] = set()
            for module_dict in self._all_injection_embedding_module_dicts():
                for embeddings in module_dict.values():
                    for embedding in embeddings:
                        if id(embedding) in seen_init:
                            continue
                        seen_init.add(id(embedding))
                        if isinstance(embedding, nn.Embedding):
                            init_dim = self.d_model
                            if module_dict is not self._injection_h_embeddings:
                                init_dim = embedding.embedding_dim
                            self.init_method.init_embeddings(
                                embedding,
                                d_model=init_dim,
                                std=self.init_std,
                                generator=generator,
                            )
                        elif hasattr(embedding, "_bucket_embedding"):
                            self.init_method.init_embeddings(
                                embedding._bucket_embedding,
                                d_model=embedding._bucket_embedding.embedding_dim,
                                std=self.init_std,
                                generator=generator,
                        )
                        if hasattr(embedding, "_scalar_weight_embeddings"):
                            for w_emb in embedding._scalar_weight_embeddings:
                                if not w_emb.weight.is_meta:
                                    nn.init.constant_(w_emb.weight, 4.0)
                        elif isinstance(embedding, EngramModule):
                            for ngram_emb in embedding.ngram_embeddings:
                                self.init_method.init_embeddings(
                                    ngram_emb,
                                    d_model=ngram_emb.embedding_dim,
                                    std=self.init_std,
                                    generator=generator,
                                )
                            self.init_method._init_linear(
                                embedding.value_proj,
                                std=self.init_std,
                                generator=generator,
                            )
                            for kp in embedding.key_projs:
                                self.init_method._init_linear(
                                    kp,
                                    std=self.init_std,
                                    generator=generator,
                                )
                            if embedding.short_conv is not None:
                                conv = embedding.short_conv.conv
                                w = conv.weight
                                local_w = w.to_local() if hasattr(w, "to_local") else w
                                if not local_w.is_meta:
                                    local_w.zero_()
                                    local_w[:, :, -1] = 1.0
        if self._embedding_injection_config is not None:
            injection_version = self._injection_version
            for block_key, gates in self._injection_h_gates.items():
                defaults = self._injection_h_gate_defaults.get(block_key, [])
                for gate_index, gate in enumerate(gates):
                    if gate_index < len(defaults):
                        gate.fill_(defaults[gate_index])
            for gate_dict in (
                self._injection_qk_gates,
                self._injection_q_gates,
                self._injection_k_gates,
                self._injection_v_gates,
                self._injection_o_gates,
            ):
                if gate_dict is None:
                    continue
                for gates in gate_dict.values():
                    for gate in gates:
                        if not gate.is_meta:
                            gate.data.fill_(self._injection_lambda_init)
            if injection_version == "Retoken":
                init_retoken_modules(self, generator=generator)
            if injection_version == "Mort":
                init_mort_modules(self, generator=generator)

        if self._shortconv_enabled:
            def _reset_conv1d_identity(conv: nn.Conv1d) -> None:
                weight = conv.weight
                local = weight.to_local() if hasattr(weight, "to_local") else weight
                local.zero_()
                local[:, :, -1] = 1.0

            # Re-init shortconv kernels to identity after to_empty()/reset_parameters.
            for conv_group in (
                self._injection_h_shortconvs.values(),
                *[module_dict.values() for module_dict in self._qkvo_shortconv_module_dicts()],
            ):
                for convs in conv_group:
                    for conv in convs:
                        if isinstance(conv, SwiGLUShortConv):
                            _reset_conv1d_identity(conv.conv_content)
                            nn.init.normal_(conv.conv_gate.weight, mean=0.0, std=self.init_std)
                            if conv.conv_gate.bias is not None:
                                conv.conv_gate.bias.data.fill_(self._sc_swiglu_gate_bias)
                        else:
                            if not conv.weight.is_meta:
                                _reset_conv1d_identity(conv)

        seen_weight_as: Set[int] = set()
        for block in self.blocks.values():
            # This might fail if it's wrapped.
            #  assert isinstance(block, TransformerBlock)
            block = cast(TransformerBlock, block)
            att = cast(Union[Attention, FusedAttention], block.attention)

            # Attention weights.
            self.init_method.init_attention(
                att,
                d_model=self.d_model,
                block_idx=block.block_idx,
                num_blocks=self.n_layers,
                std=self.init_std,
                generator=generator,
            )

            # Feed-forward weights.
            if hasattr(block, "feed_forward"):
                self.init_method.init_feed_forward(
                    block.feed_forward,
                    d_model=self.d_model,
                    block_idx=block.block_idx,
                    num_blocks=self.n_layers,
                    std=self.init_std,
                    generator=generator,
                )

            # MoE weights.
            if hasattr(block, "feed_forward_moe"):
                block = cast(MoETransformerBlock, block)
                if max_local_microbatch_size is not None:
                    block.feed_forward_moe.warmup_cache(max_local_microbatch_size)
                self.init_method.init_feed_forward_moe(
                    block.feed_forward_moe,
                    d_model=self.d_model,
                    block_idx=block.block_idx,
                    num_blocks=self.n_layers,
                    std=self.init_std,
                    generator=generator,
                )

            # Warm up RoPE cache.
            if max_seq_len is not None and att.rope is not None:
                att.rope.warmup_cache(max_seq_len, device)

        if self.lm_head is not None:
            self.init_method.init_final_w_out(
                self.lm_head.w_out,
                d_model=self.d_model,
                std=self.init_std,
                generator=generator,
            )

        # Eagerly build dense hash tables for hash injection modules to avoid compile-time graph breaks.
        if self._hash_injection_enabled:
            for module_dict in self._all_injection_embedding_module_dicts():
                for block_key, modules in module_dict.items():
                    prefix = "Output" if module_dict is self._injection_o_embeddings else "Hash"
                    for module in modules:
                        if not hasattr(module, "_ensure_dense") or not hasattr(module, "_bucket_embedding"):
                            continue
                        try:
                            weight = module._bucket_embedding.weight
                        except Exception:
                            continue
                        if weight.is_meta:
                            continue
                        target_device = weight.device
                        try:
                            log.info(
                                "Eagerly precomputing %s tables for block %s on %s",
                                prefix,
                                block_key,
                                target_device,
                            )
                            module._ensure_dense(target_device)
                        except Exception as exc:
                            log.warning(
                                "%s dense precompute failed for block %s: %s",
                                prefix,
                                block_key,
                                exc,
                            )

        return generator


    @staticmethod
    def _get_tensor_storage_elems(tensor: torch.Tensor) -> int:
        try:
            storage = tensor.untyped_storage()
            return storage.nbytes() // tensor.element_size()
        except Exception:
            try:
                return tensor.storage().size()
            except Exception:
                return 0

    def _prepare_inputs(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        *,
        ignore_index: int = -100,
        loss_reduction: Literal["mean", "sum", "none"] = "mean",
        z_loss_multiplier: Optional[float] = None,
        loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
        return_logits: Optional[bool] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> Tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        Dict[str, Any],
        Dict[int, Dict[str, Any]],
        Dict[str, Any],
    ]:
        # NOTE: with pipeline parallelism input_ids might actually be an intermediate output,
        # so we have to be careful here.
        B, S = input_ids.shape[:2]

        all_block_kwargs: Dict[str, Any] = {}
        per_block_kwargs: Dict[int, Dict[str, Any]] = defaultdict(dict)
        lm_head_kwargs: Dict[str, Any] = dict(
            ignore_index=ignore_index,
            loss_reduction=loss_reduction,
            z_loss_multiplier=z_loss_multiplier,
            return_logits=return_logits,
            logits_to_keep=logits_to_keep,
        )

        if loss_div_factor is not None:
            loss_div_factor = move_to_device(loss_div_factor, self.device)
            lm_head_kwargs["loss_div_factor"] = loss_div_factor
            all_block_kwargs["loss_div_factor"] = loss_div_factor

        # Prepare document length inputs.
        max_doc_len: Optional[int] = None
        cu_doc_lens: Optional[torch.Tensor] = None
        doc_lens: Optional[torch.Tensor] = None
        cache_leftpad: Optional[torch.Tensor] = kwargs.pop("cache_leftpad", None)

        if (doc_lens := kwargs.pop("doc_lens", None)) is not None and (
            max_doc_lens := kwargs.pop("max_doc_lens", None)
        ) is not None:
            max_doc_len = max(max_doc_lens)
            cu_doc_lens = get_cumulative_document_lengths(doc_lens)

        # Shard inputs and RoPE buffers on sequence dimension if using context parallelism.
        if (cp_load_balancer := self._cp_load_balancer) is not None:
            inputs = [input_ids]
            seq_dims = [1]
            pad_values: List[Union[int, float]] = [0]
            keys = ["input_ids"]

            # NOTE: initialize buffer(s) on CPU to avoid possible host-device sync when sharding.
            for block_idx, rope_buffers in self.get_rope_buffers(S, torch.device("cpu")).items():
                if rope_buffers is not None:
                    if rope_buffers.pos_sin is not None:
                        inputs.append(rope_buffers.pos_sin)
                        seq_dims.append(0)
                        pad_values.append(0.0)
                        keys.append(f"block_{block_idx}.pos_sin")
                    if rope_buffers.pos_cos is not None:
                        inputs.append(rope_buffers.pos_cos)
                        seq_dims.append(0)
                        pad_values.append(0.0)
                        keys.append(f"block_{block_idx}.pos_cos")
                    if rope_buffers.freqs_cis is not None:
                        inputs.append(rope_buffers.freqs_cis)
                        seq_dims.append(0)
                        pad_values.append(0.0)
                        keys.append(f"block_{block_idx}.freqs_cis")

            if labels is not None:
                inputs.append(labels)
                seq_dims.append(1)
                pad_values.append(ignore_index)
                keys.append("labels")

            if cache_leftpad is not None:
                raise NotImplementedError("cache_leftpad is not supported with context parallelism")

            if cu_doc_lens is not None:
                # NOTE: Can only shard properly here if 'input_ids' is flat, i.e. a single instance.
                # TODO: (epwalsh) We could just flatten all of the inputs here, but then we risk going
                # beyond the model's maximum sequence length, which might be okay at least
                # with relative positional encodings, but then again if you're resorting to context
                # parallelism you can probably only fit a single instance at a time anyway.
                if B != 1:
                    raise RuntimeError(
                        f"Rank micro-batches must consist of a single instance when using "
                        f"context parallelism with intra-document masking (got {B} instances)"
                    )
                inputs, additional_inputs = cp_load_balancer.batch_shard_by_document(
                    inputs=inputs,
                    seq_dims=seq_dims,
                    cu_doc_lens=cu_doc_lens,
                    pad_values=pad_values,
                    length_multiple=16,
                )
                for key, value in additional_inputs.items():
                    all_block_kwargs[key] = move_to_device(value, self.device)

            else:
                inputs = cp_load_balancer.batch_shard(
                    inputs=inputs,
                    seq_dims=seq_dims,
                    pad_values=pad_values,
                )

            for key, value in zip(keys, inputs):
                if key.startswith("block_"):
                    block_key, key = key.split(".", 1)
                    block_idx = int(block_key.replace("block_", ""))
                    per_block_kwargs[block_idx][key] = move_to_device(value, self.device)
                else:
                    all_block_kwargs[key] = move_to_device(value, self.device)

            input_ids = all_block_kwargs.pop("input_ids")
            labels = all_block_kwargs.pop("labels", None)
        else:
            input_ids = move_to_device(input_ids, self.device)
            labels = move_to_device(labels, self.device)

            if (max_doc_len is not None or cu_doc_lens is not None) and cache_leftpad is not None:
                raise ValueError("max_doc_len/cu_doc_lens and cache_leftpad are mutually exclusive")
            if max_doc_len is not None or cu_doc_lens is not None:
                all_block_kwargs["max_doc_len"] = max_doc_len
                all_block_kwargs["cu_doc_lens"] = move_to_device(cu_doc_lens, self.device)
            if cache_leftpad is not None:
                all_block_kwargs["cache_leftpad"] = move_to_device(cache_leftpad, self.device)

        return (
            input_ids,
            labels,
            all_block_kwargs,
            per_block_kwargs,
            lm_head_kwargs,
        )

    def _get_active_wandb(
        self,
        *,
        missing_attr: str,
        inactive_attr: str,
        missing_message: str,
        inactive_message: str,
    ) -> Optional[Any]:
        try:
            import wandb  # type: ignore
        except Exception:
            if not getattr(self, missing_attr, False):
                log.warning(missing_message)
                setattr(self, missing_attr, True)
            return None

        if getattr(wandb, "run", None) is None:
            if not getattr(self, inactive_attr, False):
                log.warning(inactive_message)
                setattr(self, inactive_attr, True)
            return None

        return wandb

    def _log_injection_metrics(
        self,
        *,
        h_prev: torch.Tensor,
        injection_delta: torch.Tensor,
        gate: torch.Tensor,
        input_embedding: torch.Tensor,
        layer_idx: int,
        step: Optional[int],
        lambda_raw: Optional[torch.Tensor] = None,
        log_interval: int = 50,
        eps: float = 1e-6,
        warmup_scale: float = 1.0,
    ) -> None:
        """
        Record injection diagnostics to WandB every ``log_interval`` steps on rank 0.
        """
        if step is None:
            return
        log_interval = self._injection_log_interval
        if log_interval <= 0 or step % log_interval != 0:
            return
        if self._injection_logged_steps.get(layer_idx) == step:
            return
        if dist.is_available() and dist.is_initialized():
            try:
                if dist.get_rank() != 0:
                    return
            except Exception:
                return
        wandb = self._get_active_wandb(
            missing_attr="_log_injection_wandb_missing",
            inactive_attr="_log_injection_wandb_inactive",
            missing_message="wandb not available; skipping injection metrics logging",
            inactive_message="wandb run is not initialized; skipping injection metrics logging",
        )
        if wandb is None:
            return

        with torch.no_grad():
            h_flat = h_prev.detach().reshape(-1, h_prev.shape[-1]).float()
            inj_flat = injection_delta.detach().reshape(-1, injection_delta.shape[-1]).float()
            if h_flat.shape[-1] != inj_flat.shape[-1]:
                return
            h_with_inj_flat = (h_prev + injection_delta).detach().reshape(-1, h_prev.shape[-1]).float()
            h_norm = torch.norm(h_flat, p=2, dim=-1).mean()
            inj_norm = torch.norm(inj_flat, p=2, dim=-1).mean()
            raw_irr = inj_norm / (h_norm + eps)
            gamma_val = gate.detach().reshape(-1).float()
            gamma_scalar = gamma_val[0] if gamma_val.numel() > 0 else torch.tensor(0.0)
            cos_sim = F.cosine_similarity(h_flat, inj_flat, dim=-1).mean()
            input_ok = (
                input_embedding is not None
                and input_embedding.shape[-1] == h_prev.shape[-1]
                and input_embedding.shape[-1] == injection_delta.shape[-1]
            )
            cos_inj_input = None
            cos_h_input = None
            cos_hplusinj_input = None
            if input_ok:
                inp_flat = input_embedding.detach().reshape(-1, input_embedding.shape[-1]).float()
                cos_inj_input = F.cosine_similarity(inj_flat, inp_flat, dim=-1).mean()
                cos_h_input = F.cosine_similarity(h_flat, inp_flat, dim=-1).mean()
                cos_hplusinj_input = F.cosine_similarity(h_with_inj_flat, inp_flat, dim=-1).mean()
            lambda_raw_scalar: Optional[torch.Tensor] = None
            if lambda_raw is not None:
                lambda_flat = lambda_raw.detach().reshape(-1).float()
                lambda_raw_scalar = lambda_flat[0] if lambda_flat.numel() > 0 else torch.tensor(0.0)

            payload = {
                f"analysis/layer_{layer_idx}/h_norm": h_norm.item(),
                f"analysis/layer_{layer_idx}/inj_norm": inj_norm.item(),
                f"analysis/layer_{layer_idx}/irr_raw": raw_irr.item(),
                f"analysis/layer_{layer_idx}/cos_sim": cos_sim.item(),
                f"analysis/layer_{layer_idx}/gamma": gamma_scalar.item(),
            }
            if cos_inj_input is not None:
                payload[f"analysis/layer_{layer_idx}/cos_inj_input"] = cos_inj_input.item()
            if cos_h_input is not None:
                payload[f"analysis/layer_{layer_idx}/cos_h_input"] = cos_h_input.item()
            if cos_hplusinj_input is not None:
                payload[f"analysis/layer_{layer_idx}/cos_hplusinj_input"] = cos_hplusinj_input.item()
            if lambda_raw_scalar is not None:
                payload[f"analysis/layer_{layer_idx}/lambda_raw"] = lambda_raw_scalar.item()
                payload[f"analysis/layer_{layer_idx}/lambda_after_warmup"] = (
                    lambda_raw_scalar.item() * _warmup_scale_to_python_float(warmup_scale)
                )

            wandb.log(payload, step=step)

    def _log_baseline_metrics(
        self,
        *,
        h_prev: torch.Tensor,
        input_embedding: torch.Tensor,
        layer_idx: int,
        step: Optional[int],
        log_interval: int = 50,
        eps: float = 1e-6,
    ) -> None:
        """
        Record baseline (no-injection) diagnostics to WandB every ``log_interval`` steps on rank 0.
        """
        if step is None:
            return
        log_interval = self._injection_log_interval
        if log_interval <= 0 or step % log_interval != 0:
            return
        if self._baseline_logged_steps.get(layer_idx) == step:
            return
        if dist.is_available() and dist.is_initialized():
            try:
                if dist.get_rank() != 0:
                    return
            except Exception:
                return
        wandb = self._get_active_wandb(
            missing_attr="_log_baseline_wandb_missing",
            inactive_attr="_log_baseline_wandb_inactive",
            missing_message="wandb not available; skipping baseline metrics logging",
            inactive_message="wandb run is not initialized; skipping baseline metrics logging",
        )
        if wandb is None:
            return

        with torch.no_grad():
            h_flat = h_prev.detach().reshape(-1, h_prev.shape[-1]).float()
            inp_flat = input_embedding.detach().reshape(-1, input_embedding.shape[-1]).float()
            h_norm = torch.norm(h_flat, p=2, dim=-1).mean()
            input_norm = torch.norm(inp_flat, p=2, dim=-1).mean()
            cos_h_input = F.cosine_similarity(h_flat, inp_flat, dim=-1).mean()

            payload = {
                f"analysis/layer_{layer_idx}/h_norm": h_norm.item(),
                f"analysis/layer_{layer_idx}/input_norm": input_norm.item(),
                f"analysis/layer_{layer_idx}/cos_h_input": cos_h_input.item(),
            }

            wandb.log(payload, step=step)
            self._baseline_logged_steps[layer_idx] = step
            self._injection_logged_steps[layer_idx] = step

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        labels: Optional[torch.Tensor] = None,
        ignore_index: int = -100,
        loss_reduction: Literal["mean", "sum", "none"] = "mean",
        z_loss_multiplier: Optional[float] = None,
        loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
        return_logits: Optional[bool] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        step: Optional[int] = None,
        **kwargs,
    ) -> Union[torch.Tensor, LMOutputWithLoss]:
        """
        Run the transformer on the token input IDs.

        :param input_ids: The token input IDs, shape ``(batch_size, seq_len)``.
        :param labels: The token labels, shape ``(batch_size, seq_len)``.
        :param ignore_index: The index to ignore in the loss computation. Default is -100.
        :param loss_reduction: The reduction method for the loss. Can be "mean", "sum", or "none".
        :param z_loss_multiplier: Optional multiplier for the z-loss regularization term.
        :param loss_div_factor: Optional divisor for the loss, can be a scalar or tensor.
        :param return_logits: Whether to return logits along with the loss when labels are provided.
        :param logits_to_keep: Number of positions to keep from the end of the sequence (if int),
            or tensor specifying which positions to keep. Default is 0 (keep all).

        :returns: The logits if ``labels`` is ``None`` or the losses if ``labels`` is not ``None``.
        """
        # Allow callers to pass step either as kwarg or positional to support logging.
        if step is None:
            step = kwargs.pop("step", None)
        else:
            # Avoid passing step further down to blocks.
            kwargs.pop("step", None)
        (
            input_ids,
            labels,
            all_block_kwargs,
            per_block_kwargs,
            lm_head_kwargs,
        ) = self._prepare_inputs(
            input_ids,
            labels,
            ignore_index=ignore_index,
            loss_reduction=loss_reduction,
            z_loss_multiplier=z_loss_multiplier,
            loss_div_factor=loss_div_factor,
            return_logits=return_logits,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )

        # Get embeddings but pass-through for non-existent layers to allow easy
        # pipeline parallel configuration.
        h = self.embeddings(input_ids) if self.embeddings is not None else input_ids
        input_embedding = h
        # Use a detached clone for logging to avoid Dynamo seeing aliases to the block input.
        log_input_embedding = input_embedding.view_as(input_embedding).detach()

        # Resolve the effective injection family/targets once per forward.
        injection_version = self._injection_version
        injection_targets = list(getattr(self, "_injection_targets", []))
        h_target_enabled = "h" in injection_targets
        attention_has_qkv = any(
            t in {"q", "k", "v"} for t in getattr(self, "_attention_injection_targets", [])
        )

        # Run each block.
        for block_key, block in self.blocks.items():
            block_idx = int(block_key)
            block_kwargs = per_block_kwargs.get(block_idx, {})
            block_kwargs["_injection_version"] = injection_version
            warmup_scale = 1.0
            if (
                self._injection_lambda_warmup_enabled
                and step is not None
                and self._injection_lambda_warmup_steps > 0
            ):
                warmup_scale = min(
                    1.0,
                    float(step) / float(self._injection_lambda_warmup_steps),
                )
            warmup_scale_tensor = torch.tensor(
                warmup_scale,
                dtype=h.dtype,
                device=h.device,
            )

            if injection_version == "None":
                self._log_baseline_metrics(
                    h_prev=h,
                    input_embedding=input_embedding,
                    layer_idx=block_idx,
                    step=step,
                )

            context = InjectionBlockContext(
                block_key=block_key,
                block_idx=block_idx,
                step=step,
                input_ids=input_ids,
                input_embedding=log_input_embedding,
                hidden_states=h,
            )
            if injection_version == "Retoken":
                retoken_result = prepare_retoken_block_kwargs(self, context)
                block_kwargs.update(retoken_result.block_kwargs)
            elif injection_version == "Engram":
                h = apply_engram_pre_block(self, context)
            elif injection_version == "Mort":
                mort_result = prepare_mort_block_kwargs(self, context)
                block_kwargs.update(mort_result.block_kwargs)
            elif injection_version == "X-gram":
                xgram_result = prepare_xgram_block_kwargs(
                    self,
                    context,
                    warmup_scale_tensor=warmup_scale_tensor,
                )
                h = xgram_result.hidden_states
                block_kwargs.update(xgram_result.block_kwargs)

            # Normalize block_kwargs: fixed-order template ensures every block
            # receives the same key set and key order, which keeps torch.compile
            # graphs and activation-checkpoint recomputation consistent across
            # injection and non-injection layers.
            _uniform_block_kwargs = {
                "_injection_h_embeddings": None,
                "_injection_h_gates": None,
                "_injection_qk_delta": None,
                "_injection_q_delta": None,
                "_injection_k_delta": None,
                "_injection_v_delta": None,
                "_injection_warmup_scale": None,
                "_injection_version": "None",
                "_injection_sc_rmsnorm_eps": 1e-5,
                "_injection_targets": (),
                "input_ids": None,
                "_injection_o_delta": None,
                "_retoken_embeddings": None,
                "_retoken_scalers": None,
            }
            _uniform_block_kwargs.update(block_kwargs)
            block_kwargs = _uniform_block_kwargs

            # Mark sizes as dynamic for torch.compile().
            if self.compile_enabled:
                mark_dynamic(h, (0, 1), strict=False)
            h = block(h, **all_block_kwargs, **block_kwargs)

        # Get final logits but again pass-through in case of pipeline parallelism.
        if self.lm_head is not None:
            if self.compile_enabled:
                mark_dynamic(h, (0, 1), strict=False)
                if labels is not None:
                    mark_dynamic(labels, (0, 1), strict=False)
            # NOTE: When TP is active we can't pass 'labels=None' or the hook from 'PrepareModuleInput'
            # will throw an exception.
            if labels is not None:
                lm_head_kwargs["labels"] = labels
            return self.lm_head(h, **lm_head_kwargs)
        else:
            return h

    def apply_fp8(self, float8_config: Float8Config):
        """
        Use an FP8 recipe on most linear layers.
        """
        if not float8_config.enabled:
            return

        modules_to_ignore = set()
        if self.lm_head is not None:
            modules_to_ignore.add("lm_head.w_out")

        float8_config.apply_float8_linear(self, modules_to_ignore=modules_to_ignore)

        self._fp8_enabled = True
        self._precompute_float8_dynamic_scale_for_fsdp = (
            float8_config.should_precompute_float8_dynamic_scale_for_fsdp
        )

    def apply_pp(self, pp_mesh: DeviceMesh):
        """
        Prepare the model for pipeline parallelism after it's been split into stages.
        """
        for block in self.blocks.values():
            block = cast(TransformerBlockBase, block)
            block.apply_pp(pp_mesh)
        self._pp_enabled = True
        self._pp_group_size = pp_mesh.size()

    def apply_tp(self, tp_mesh: DeviceMesh, float8_enabled: Optional[bool] = None):
        """
        Apply tensor parallelism to the model.

        :param loss_parallel: Set to ``True`` if parallelizing the loss function as well.
        :param float8_enabled: Set this to ``True`` if training with float8 linear layers.
        """
        if float8_enabled is None:
            float8_enabled = self.fp8_enabled
        elif not float8_enabled and self.fp8_enabled:
            raise OLMoConfigurationError(
                "Got 'float8_enabled=False', but FP8 has already been enabled"
            )

        if self.embeddings is not None:
            parallelize_module(
                self.embeddings,
                device_mesh=tp_mesh,
                parallelize_plan=RowwiseParallel(
                    input_layouts=Replicate(),
                    output_layouts=Shard(1),
                    use_local_output=False,
                ),
            )
        if self._engram_enabled:
            for modules in self._injection_h_embeddings.values():
                for module in modules:
                    if isinstance(module, EngramModule):
                        for ngram_emb in module.ngram_embeddings:
                            parallelize_module(
                                ngram_emb,
                                device_mesh=tp_mesh,
                                parallelize_plan=RowwiseParallel(
                                    input_layouts=Replicate(),
                                    output_layouts=Shard(1),
                                    use_local_output=False,
                                ),
                            )
                        # value_proj and key_projs: RowwiseParallel on output dim
                        parallelize_module(
                            module.value_proj,
                            device_mesh=tp_mesh,
                            parallelize_plan=RowwiseParallel(
                                input_layouts=Replicate(),
                                output_layouts=Shard(1),
                                use_local_output=False,
                            ),
                        )
                        for kp in module.key_projs:
                            parallelize_module(
                                kp,
                                device_mesh=tp_mesh,
                                parallelize_plan=RowwiseParallel(
                                    input_layouts=Replicate(),
                                    output_layouts=Shard(1),
                                    use_local_output=False,
                                ),
                            )
        seen_tp: Set[int] = set()
        for module_dict in (
            self._injection_h_embeddings,
            self._injection_qk_embeddings,
            self._injection_q_embeddings,
            self._injection_k_embeddings,
            self._injection_v_embeddings,
        ):
            if module_dict is None:
                continue
            for embeddings in module_dict.values():
                for embedding in embeddings:
                    if id(embedding) in seen_tp:
                        continue
                    seen_tp.add(id(embedding))
                    if isinstance(embedding, nn.Embedding):
                        parallelize_module(
                            embedding,
                            device_mesh=tp_mesh,
                            parallelize_plan=RowwiseParallel(
                                input_layouts=Replicate(),
                                output_layouts=Shard(1),
                                use_local_output=False,
                            ),
                        )
                    elif hasattr(embedding, "_bucket_embedding"):
                        parallelize_module(
                            embedding._bucket_embedding,
                            device_mesh=tp_mesh,
                            parallelize_plan=RowwiseParallel(
                                input_layouts=Replicate(),
                                output_layouts=Shard(1),
                                use_local_output=False,
                            ),
                        )
                        if hasattr(embedding, "_scalar_weight_embeddings"):
                            for w_emb in embedding._scalar_weight_embeddings:
                                parallelize_module(
                                    w_emb,
                                    device_mesh=tp_mesh,
                                    parallelize_plan=RowwiseParallel(
                                        input_layouts=Replicate(),
                                        output_layouts=Shard(1),
                                        use_local_output=False,
                                    ),
                                )
        for conv_dict in (
            self._injection_h_shortconvs,
            self._injection_qk_shortconvs,
            self._injection_q_shortconvs,
            self._injection_k_shortconvs,
            self._injection_v_shortconvs,
        ):
            if conv_dict is None:
                continue
            for convs in conv_dict.values():
                for conv in convs:
                    if id(conv) in seen_tp:
                        continue
                    seen_tp.add(id(conv))
                    targets = (
                        [conv.conv_content, conv.conv_gate]
                        if isinstance(conv, SwiGLUShortConv)
                        else [conv]
                    )
                    for sub_conv in targets:
                        parallelize_module(
                            sub_conv,
                            device_mesh=tp_mesh,
                            parallelize_plan=RowwiseParallel(
                                input_layouts=Shard(1),
                                output_layouts=Shard(1),
                                use_local_output=False,
                            ),
                        )

        # O injection modules: keep replicate semantics to match w_out output accumulation.
        if self._injection_o_embeddings:
            for embeddings in self._injection_o_embeddings.values():
                for embedding in embeddings:
                    parallelize_module(
                        embedding,
                        device_mesh=tp_mesh,
                        parallelize_plan=ColwiseParallel(
                            input_layouts=Replicate(),
                            output_layouts=Replicate(),
                            use_local_output=True,
                        ),
                    )
        if self._injection_o_shortconvs:
            for convs in self._injection_o_shortconvs.values():
                for conv in convs:
                    targets = (
                        [conv.conv_content, conv.conv_gate]
                        if isinstance(conv, SwiGLUShortConv)
                        else [conv]
                    )
                    for sub_conv in targets:
                        parallelize_module(
                            sub_conv,
                            device_mesh=tp_mesh,
                            parallelize_plan=SequenceParallel(
                                input_layouts=Replicate(),
                                output_layouts=Replicate(),
                                use_local_output=True,
                            ),
                        )
        for embeddings in self._retoken_embeddings.values():
            for embedding in embeddings:
                parallelize_module(
                    embedding,
                    device_mesh=tp_mesh,
                    parallelize_plan=RowwiseParallel(
                        input_layouts=Replicate(),
                        output_layouts=Shard(1),
                        use_local_output=False,
                    ),
                )

        # Apply tensor/sequence parallelism to every transformer block.
        for block in self.blocks.values():
            block = cast(TransformerBlockBase, block)
            block.apply_tp(tp_mesh, input_layout=Shard(1), float8_enabled=float8_enabled)

        if self.lm_head is not None:
            self.lm_head.apply_tp(tp_mesh, input_layouts=(Shard(1), Replicate()))

        self._tp_enabled = True
        self._tp_mesh = tp_mesh

    def apply_cp(
        self,
        cp_mesh: DeviceMesh,
        load_balancer: RingAttentionLoadBalancerType,
        head_stride: int = 1,
    ):
        """
        Prepare the model for context-parallelism (CP).

        :param cp_mesh: The CP device mesh.
        :param load_balancer: The load balancing method.
        """
        self._cp_load_balancer = load_balancer.build(cp_mesh)
        for block in self.blocks.values():
            cast(TransformerBlockBase, block).apply_cp(
                cp_mesh, load_balancer, head_stride=head_stride
            )
        if self.lm_head is not None:
            self.lm_head.apply_cp(cp_mesh, load_balancer)

    def apply_activation_checkpointing(
        self,
        mode: TransformerActivationCheckpointingMode,
        block_interval: Optional[int] = None,
        modules: Optional[List[str]] = None,
        activation_memory_budget: Optional[float] = None,
    ):
        """
        Apply activation checkpointing to the model.

        :param mode: Determines how to apply activation checkpointing.
        :param block_interval: Required when :data:`mode` is "selected_blocks". Determines
            which blocks are wrapped.
        :param modules: Required when :data:`mode` is "selected_modules". A list of modules names
            to wrap for activation checkpointing. Globs are supported.
        :param activation_memory_budget: The memory budget for activation checkpointing in the range
            [0, 1]. 0 corresponds to the memory usage when recomputing all activations, and 1
            corresponds to the memory usage when recomputing no activations (which is the default).
            Requires compilation to be enabled.
        """

        if mode == TransformerActivationCheckpointingMode.budget:
            if activation_memory_budget is None:
                raise ValueError("'activation_memory_budget' is required for 'budget' mode")
            if activation_memory_budget < 0 or activation_memory_budget > 1:
                raise ValueError("'activation_memory_budget' must be in the range [0, 1]")
            torch._functorch.config.activation_memory_budget = activation_memory_budget
            return

        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            CheckpointImpl,
            checkpoint_wrapper as ptd_checkpoint_wrapper,
        )

        if (
            mode == TransformerActivationCheckpointingMode.selected_blocks
            and block_interval is None
        ):
            raise ValueError("'block_interval' is required for 'selected_blocks' mode")

        if mode == TransformerActivationCheckpointingMode.selected_modules and modules is None:
            raise ValueError("'modules' is required for 'selected_modules' mode")

        # TODO: only preserve RNG state if dropout is active
        preserve_rng_state = False
        # Use non-reentrant checkpointing so kwargs (e.g., injection args) safely propagate.
        checkpoint_impl = CheckpointImpl.NO_REENTRANT

        if mode == TransformerActivationCheckpointingMode.selected_modules:
            from fnmatch import fnmatch

            assert modules is not None
            wrapped_modules: Set[str] = set()
            for name, module in self.named_modules():
                for pattern in modules:
                    if fnmatch(name, pattern):
                        break
                else:
                    continue

                if isinstance(module, MoEBase):
                    raise OLMoConfigurationError(
                        "Wrapping an entire MoE module for activation checkpointing is not supported. "
                        "Please try a finer-grained wrapping strategy."
                    )

                # NOTE: have to be careful not to try to wrap submodules of modules that have been wrapped.
                parent_name = ".".join(name.split(".")[:-1])
                if parent_name in wrapped_modules:
                    continue

                parent = self if not parent_name else self.get_submodule(parent_name)
                module = ptd_checkpoint_wrapper(
                    module,
                    preserve_rng_state=preserve_rng_state,
                    checkpoint_impl=checkpoint_impl,
                )
                parent.register_module(name.split(".")[-1], module)
                log.info(f"Wrapped '{name}' for activation checkpointing")
                wrapped_modules.add(name)
        else:
            for block_idx, block in enumerate(self.blocks.values()):
                if mode == TransformerActivationCheckpointingMode.selected_blocks:
                    assert block_interval is not None
                    if block_idx % block_interval == 0:
                        if isinstance(block, MoETransformerBlock):
                            raise OLMoConfigurationError(
                                "Wrapping MoE blocks for activation checkpointing is not supported."
                            )
                        block = ptd_checkpoint_wrapper(
                            block,
                            preserve_rng_state=preserve_rng_state,
                            checkpoint_impl=checkpoint_impl,
                        )
                elif mode == TransformerActivationCheckpointingMode.full:
                    if isinstance(block, MoETransformerBlock):
                        raise OLMoConfigurationError(
                            "Wrapping MoE blocks for activation checkpointing is not supported."
                        )
                    block = ptd_checkpoint_wrapper(
                        block,
                        preserve_rng_state=preserve_rng_state,
                        checkpoint_impl=checkpoint_impl,
                    )
                elif mode == TransformerActivationCheckpointingMode.selected_ops:
                    block = ptd_checkpoint_wrapper(
                        block,
                        context_fn=selective_checkpointing_context_fn,
                        preserve_rng_state=preserve_rng_state,
                        checkpoint_impl=checkpoint_impl,
                    )

                self.blocks.register_module(str(block_idx), block)

    def apply_compile(self):
        """
        Apply ``torch.compile()`` to each transformer block, which makes compilation efficient
        due to repeated structure.

        .. warning::
            This must be called after :meth:`apply_activation_checkpointing()` but
            before :meth:`apply_fsdp()` or :meth:`apply_ddp()`.
        """
        for block in self.blocks.values():
            block = cast(TransformerBlockBase, block)
            block.apply_compile()

        if self.lm_head is not None:
            self.lm_head.compile(fullgraph=False)

        self._compile_enabled = True

    def apply_fsdp(
        self,
        dp_mesh: Optional[DeviceMesh] = None,
        param_dtype: Optional[torch.dtype] = None,
        reduce_dtype: torch.dtype = torch.float32,
        pp_enabled: bool = False,
        prefetch_factor: int = 0,
        wrapping_strategy: TransformerDataParallelWrappingStrategy = TransformerDataParallelWrappingStrategy.full,
    ):
        """
        Apply FSDP(2) to the model.

        .. warning::
            This should generally be called last if using any other parallelism strategies or optimizations
            like :meth:`apply_compile()`.

        :param dp_mesh: The model data parallel device mesh.
        :param param_dtype: The data type to materialize params in. Defaults to the current param dtype.
        :param reduce_dtype: The data type for gradient reduction.
        :pp_enabled: If pipeline parallelism is also enabled.
        :prefetch_factor: For tuning the prefetch settings. 0 is the default, and higher values result
            in more aggressive prefetching.
        :wrapping_strategy: The wrapping strategy.
        """

        mp_policy = MixedPrecisionPolicy(
            param_dtype=param_dtype or self.dtype, reduce_dtype=reduce_dtype
        )
        fsdp_config = dict(mesh=dp_mesh, mp_policy=mp_policy)
        # For PP, do not reshard after forward to avoid per-microbatch all-gathers,
        # which can be expensive and non-overlapped
        reshard_after_forward = False if pp_enabled else True

        seen_weight_as: Set[int] = set()

        for block in self.blocks.values():
            block = cast(TransformerBlockBase, block)
            block.apply_fsdp(
                dp_mesh=dp_mesh,
                prefetch_factor=prefetch_factor,
                wrapping_strategy=wrapping_strategy,
                reshard_after_forward=reshard_after_forward,
                mp_policy=mp_policy,
            )

        if self.embeddings is not None:
            fully_shard(
                self.embeddings,
                reshard_after_forward=reshard_after_forward,
                **fsdp_config,
            )
            # Embedding params are not needed for backwards computation.
            cast(FSDPModule, self.embeddings).set_unshard_in_backward(False)

        if self._engram_enabled:
            for modules in self._injection_h_embeddings.values():
                for module in modules:
                    if isinstance(module, EngramModule):
                        for ngram_emb in module.ngram_embeddings:
                            fully_shard(
                                ngram_emb,
                                reshard_after_forward=reshard_after_forward,
                                **fsdp_config,
                            )
                            cast(FSDPModule, ngram_emb).set_unshard_in_backward(False)

        for module_dict in (
            self._injection_h_embeddings,
            self._injection_qk_embeddings,
            self._injection_q_embeddings,
            self._injection_k_embeddings,
            self._injection_v_embeddings,
            self._injection_o_embeddings,
        ):
            if module_dict is None:
                continue
            for injection_embeddings in module_dict.values():
                for embedding in injection_embeddings:
                    if id(embedding) in seen_weight_as:
                        continue
                    if isinstance(embedding, nn.Embedding):
                        fully_shard(
                            embedding,
                            reshard_after_forward=reshard_after_forward,
                            **fsdp_config,
                        )
                        cast(FSDPModule, embedding).set_unshard_in_backward(False)
                        seen_weight_as.add(id(embedding))
                    elif hasattr(embedding, "_bucket_embedding"):
                        if id(embedding._bucket_embedding) in seen_weight_as:
                            continue
                        fully_shard(
                            embedding._bucket_embedding,
                            reshard_after_forward=reshard_after_forward,
                            **fsdp_config,
                        )
                        cast(FSDPModule, embedding._bucket_embedding).set_unshard_in_backward(False)
                        seen_weight_as.add(id(embedding._bucket_embedding))
                        if hasattr(embedding, "_scalar_weight_embeddings"):
                            for w_emb in embedding._scalar_weight_embeddings:
                                if id(w_emb) in seen_weight_as:
                                    continue
                                fully_shard(
                                    w_emb,
                                    reshard_after_forward=reshard_after_forward,
                                    **fsdp_config,
                                )
                                seen_weight_as.add(id(w_emb))
        for conv_dict in (
            self._injection_h_shortconvs,
            self._injection_qk_shortconvs,
            self._injection_q_shortconvs,
            self._injection_k_shortconvs,
            self._injection_v_shortconvs,
            self._injection_o_shortconvs,
        ):
            if conv_dict is None:
                continue
            for convs in conv_dict.values():
                for conv in convs:
                    if id(conv) in seen_weight_as:
                        continue
                    targets = (
                        [conv.conv_content, conv.conv_gate]
                        if isinstance(conv, SwiGLUShortConv)
                        else [conv]
                    )
                    for sub_conv in targets:
                        if id(sub_conv) in seen_weight_as:
                            continue
                        fully_shard(
                            sub_conv,
                            reshard_after_forward=reshard_after_forward,
                            **fsdp_config,
                        )
                        seen_weight_as.add(id(sub_conv))
        for retoken_embeddings in self._retoken_embeddings.values():
            for embedding in retoken_embeddings:
                fully_shard(
                    embedding,
                    reshard_after_forward=reshard_after_forward,
                    **fsdp_config,
                )
                cast(FSDPModule, embedding).set_unshard_in_backward(False)
        if (
            wrapping_strategy != TransformerDataParallelWrappingStrategy.blocks
            and self.lm_head is not None
        ):
            fully_shard(self.lm_head, reshard_after_forward=False, **fsdp_config)

        fully_shard(self, reshard_after_forward=reshard_after_forward, **fsdp_config)
        # Some inputs need to be on CPU initially, but FSDP will move everything to model's
        # device if we don't hide it.
        self.register_forward_pre_hook(_hide_cpu_inputs_from_torch, prepend=True, with_kwargs=True)
        self.register_forward_pre_hook(
            _unhide_cpu_inputs_from_torch, prepend=False, with_kwargs=True
        )

        if prefetch_factor > 0:
            blocks = cast(List[FSDPModule], list(self.blocks.values()))
            for i in range(len(blocks)):
                block = blocks[i]
                if i + 1 < len(blocks):
                    block.set_modules_to_forward_prefetch(blocks[i + 1 : i + 1 + prefetch_factor])
                elif isinstance(self.lm_head, FSDPModule):
                    block.set_modules_to_forward_prefetch([self.lm_head])

        self._fsdp_enabled = True

    def apply_ddp(
        self,
        dp_mesh: Optional[DeviceMesh] = None,
        param_dtype: Optional[torch.dtype] = None,
        compile_enabled: bool = False,
        autograd_compile_enabled: bool = False,
    ):
        """
        Apply DDP to the model.
        """
        from torch.distributed._composable.replicate import replicate

        # Cast model explicitly to the specified dtype before applying DDP
        target_dtype = param_dtype or self.dtype
        if target_dtype != self.dtype:
            self.to(dtype=target_dtype)

        # Adapted from
        # https://github.com/pytorch/torchtitan/blob/90c889e972b56b9faadebbb78fc985dedc537ed9/torchtitan/parallelisms/parallelize_llama.py#L328
        if compile_enabled:
            if autograd_compile_enabled:
                torch._dynamo.config.optimize_ddp = "python_reducer_without_compiled_forward"  # type: ignore
            else:
                torch._dynamo.config.optimize_ddp = "ddp_optimizer"  # type: ignore

        replicate(self, device_mesh=dp_mesh, bucket_cap_mb=100)
        # Some inputs need to be on CPU initially, but DDP will move everything to model's
        # device if we don't hide it.
        self.register_forward_pre_hook(_hide_cpu_inputs_from_torch, prepend=True, with_kwargs=True)
        self.register_forward_pre_hook(
            _unhide_cpu_inputs_from_torch, prepend=False, with_kwargs=True
        )

    @property
    def num_params(self) -> int:
        if self._cached_num_params is None:
            self._cached_num_params = sum(p.numel() for p in self.parameters())
        return self._cached_num_params

    @property
    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def num_non_embedding_params(self) -> int:
        if self._cached_num_non_embedding_params is None:
            params = sum(p.numel() for p in self.parameters())
            params -= _embedding_like_numel(self.embeddings)
            for module_dict in self._all_injection_embedding_module_dicts(include_retoken=True):
                for embeddings in module_dict.values():
                    for embedding in embeddings:
                        params -= _embedding_like_numel(embedding)
            self._cached_num_non_embedding_params = params
        return self._cached_num_non_embedding_params

    def num_flops_per_token(self, seq_len: int) -> int:
        """
        Get the approximate number of flops per token.
        """
        def _unwrap(module: torch.nn.Module) -> torch.nn.Module:
            for attr in ("_orig_module", "module", "_fsdp_wrapped_module"):
                inner = getattr(module, attr, None)
                if inner is not None and inner is not module:
                    return _unwrap(inner)
            return module

        base = _unwrap(self)
        n, h, q, t = (
            base.n_layers,
            base.n_attn_heads,
            base.d_model // base.n_attn_heads,
            seq_len,
        )
        base_params = getattr(base, "num_non_embedding_params", None)
        if base_params is None:
            base_params = sum(p.numel() for p in base.parameters())
        # Reasoning behind the factor of 12 for the self-attention part of the formula:
        # 1. each self-attention has 2 matmul in the forward and 4 in the backward (6)
        # 2. the flash attention does 1 more matmul recomputation in the backward
        #    but recomputation should not be counted in calculating MFU           (+0)
        # 3. each matmul performs 1 multiplication and 1 addition                 (*2)
        # 4. we follow the convention and do not account for sparsity in causal attention
        flop_per_token = 6 * base_params + 12 * n * h * q * t

        return flop_per_token

    def post_batch(self, dry_run: bool = False):
        """
        Should be called right after the final backward of a complete batch but before the optimizer step.
        """
        del dry_run

    def post_optim_step(self):
        """
        Should be called right after an optimizer step.
        """
        if self.fp8_enabled and self._precompute_float8_dynamic_scale_for_fsdp:
            from torchao.float8 import precompute_float8_dynamic_scale_for_fsdp

            precompute_float8_dynamic_scale_for_fsdp(self)


@beta_feature
class NormalizedTransformer(Transformer):
    """
    A nGPT transformer implementation, to be used with the :class:`NormalizedTransformerBlock` block
    type.
    """

    def __init__(
        self,
        *,
        d_model: int,
        vocab_size: int,
        n_layers: int,
        block: TransformerBlockConfig,
        lm_head: LMHeadConfig,
        dtype: torch.dtype = torch.float32,
        init_method: InitMethod = InitMethod.normalized,
        init_device: str = "cpu",
        init_seed: int = 0,
        init_std: float = 0.02,
        block_overrides: Optional[Dict[int, TransformerBlockConfig]] = None,
        embedding_injection: Optional[TransformerEmbeddingInjectionConfig] = None,
    ):
        super().__init__(
            d_model=d_model,
            vocab_size=vocab_size,
            n_layers=n_layers,
            block=block,
            lm_head=lm_head,
            dtype=dtype,
            init_method=init_method,
            init_device=init_device,
            init_seed=init_seed,
            init_std=init_std,
            block_overrides=block_overrides,
            embedding_injection=embedding_injection,
        )

    def _validate_block(self, block: TransformerBlockBase) -> TransformerBlockBase:
        if not isinstance(block, NormalizedTransformerBlock):
            raise OLMoConfigurationError(
                f"'{self.__class__.__name__}' requires a '{NormalizedTransformerBlock.__name__}' block"
            )
        return block

    @torch.no_grad()
    def init_weights(self, *args, **kwargs) -> torch.Generator:
        generator = super().init_weights(*args, **kwargs)
        self.normalize_matrices()
        return generator

    @torch.no_grad()
    def normalize_matrices(self):
        """
        Normalize the weights in all matrices. This should be called after each optimizer step, which
        the :class:`~olmo_core.train.train_module.TransformerTrainModule` will handle for you.
        """
        if self.embeddings is not None:
            self._normalize_matrix(self.embeddings.weight)

        for block in self.blocks.values():
            if hasattr(block, "normalize_matrices"):
                block.normalize_matrices()  # type: ignore

        if self.lm_head is not None:
            self.lm_head.normalize_matrices()  # type: ignore

    def _normalize_matrix(self, w: torch.Tensor, dim: int = -1):
        w.copy_(l2_normalize(w, dim=dim))

    def apply_tp(
        self,
        tp_mesh: DeviceMesh,
        float8_enabled: Optional[bool] = None,
    ):
        del tp_mesh, float8_enabled

        raise NotImplementedError(
            "TP is not implemented yet for the normalized transformer variant"
        )

    def apply_compile(self):
        super().apply_compile()
        self.normalize_matrices = torch.compile(self.normalize_matrices)

    def post_optim_step(self):
        super().post_optim_step()
        self.normalize_matrices()


@beta_feature
class MoETransformer(Transformer):
    """
    An MoE transformer implementation, to be used with one of the
    :class:`MoETransformerBlock` block types.
    """

    @property
    def is_moe(self) -> bool:
        return True

    def compute_auxiliary_metrics(
        self, reset: bool = True
    ) -> Dict[str, Tuple[torch.Tensor, Optional["ReduceType"]]]:
        from olmo_core.train.common import ReduceType

        mean_offset = 1.0
        if self.pp_enabled:
            # Change the divisor to 'world_size // pp_group_size'
            mean_offset = self._pp_group_size

        out: Dict[str, Tuple[torch.Tensor, Optional["ReduceType"]]] = {}
        for block_idx, block in self.blocks.items():
            if not block.is_moe:
                continue
            block = cast(MoETransformerBlock, block)
            block_metrics = block.compute_metrics(reset=reset)
            for metric_name, (metric_val, reduce_type) in block_metrics.items():
                out[f"block {int(block_idx):02d}/{metric_name}"] = (
                    metric_val,
                    reduce_type,
                )

                if self.pp_enabled and reduce_type == ReduceType.mean:
                    metric_val = metric_val.float() * mean_offset

                if metric_name not in out:
                    out[metric_name] = (metric_val, reduce_type)
                elif reduce_type in (ReduceType.mean, ReduceType.sum):
                    out[metric_name] = (
                        out[metric_name][0] + metric_val,
                        reduce_type,
                    )
                elif reduce_type == ReduceType.max:
                    out[metric_name] = (
                        torch.max(out[metric_name][0], metric_val),
                        reduce_type,
                    )
                else:
                    raise NotImplementedError(reduce_type)
        return out

    def reset_auxiliary_metrics(self):
        for block in self.blocks.values():
            if not block.is_moe:
                continue
            cast(MoETransformerBlock, block).reset_metrics()

    def apply_ep(self, ep_mesh: DeviceMesh, **kwargs):
        for block in self.blocks.values():
            if not block.is_moe:
                continue
            block = cast(MoETransformerBlock, block)
            block.apply_ep(ep_mesh, **kwargs)

    def prepare_experts_for_fsdp(
        self,
        world_mesh: DeviceMesh,
        param_dtype: Optional[torch.dtype] = None,
        reduce_dtype: torch.dtype = torch.float32,
        pp_enabled: bool = False,
    ):
        for block in self.blocks.values():
            if not block.is_moe:
                continue
            block = cast(MoETransformerBlock, block)
            reshard_after_forward = True
            if pp_enabled or block.ep_enabled or block.tp_enabled:
                reshard_after_forward = False
            block.feed_forward_moe.prepare_experts_for_fsdp(
                world_mesh=world_mesh,
                mp_policy=MixedPrecisionPolicy(
                    param_dtype=param_dtype or self.dtype, reduce_dtype=reduce_dtype
                ),
                reshard_after_forward=reshard_after_forward,
            )

    def prepare_experts_for_ddp(self, world_mesh: DeviceMesh):
        for block in self.blocks.values():
            if not block.is_moe:
                continue
            cast(MoETransformerBlock, block).feed_forward_moe.prepare_experts_for_ddp(
                world_mesh=world_mesh,
            )

def post_batch(self, dry_run: bool = False):
        for block in self.blocks.values():
            if not block.is_moe:
                continue
            block = cast(MoETransformerBlock, block)
            block.feed_forward_moe.post_batch(dry_run=dry_run)


def _hide_cpu_inputs_from_torch(m, args, kwargs) -> Optional[Tuple[Any, Dict[str, Any]]]:
    del m
    if (doc_lens := kwargs.get("doc_lens")) is not None:
        kwargs["doc_lens"] = hide_from_torch(doc_lens)
    return (args, kwargs)


def _unhide_cpu_inputs_from_torch(m, args, kwargs) -> Optional[Tuple[Any, Dict[str, Any]]]:
    del m
    if (doc_lens := kwargs.get("doc_lens")) is not None:
        kwargs["doc_lens"] = unhide_from_torch(doc_lens)
    return (args, kwargs)
