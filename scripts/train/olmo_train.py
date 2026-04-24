import argparse
import glob
import math
import os
import socket
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Literal
import gc
import time
import json
import torch
import torch.nn as nn
import torch.distributed as dist
import logging
from omegaconf import OmegaConf

# Default W&B endpoints/project; allow override via env
os.environ.setdefault("WANDB_API_KEY", "")
os.environ.setdefault("WANDB_BASE_URL", "")
os.environ.setdefault("WANDB_PROJECT", "")
os.environ.setdefault("WANDB_ENTITY", "")
os.environ.setdefault("WANDB_MODE", "online")


from olmo_core.config import Config, DType
from olmo_core.data import (
    TokenizerConfig,
    UBDataLoaderConfig,
    NumpyDatasetConfig,
    NumpyDatasetType,
)
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.transformer import Transformer
from olmo_core.nn.transformer.config import TransformerConfig, TransformerEmbeddingInjectionConfig
from olmo_core.nn.lm_head import LMOutputWithLoss
from olmo_core.nn.feed_forward import FeedForwardConfig
from olmo_core.optim import CosWithWarmup, OptimGroupOverride, SkipStepAdamWConfig
from olmo_core.train import (
    Duration,
    LoadStrategy,
    TrainerConfig,
    prepare_training_environment,
    teardown_training_environment,
)
from olmo_core.train.callbacks import (
    AveragingEvaluatorCallback,
    CheckpointerCallback,
    CometCallback,
    ConfigSaverCallback,
    DownstreamEvaluatorCallbackConfig,
    FilteredDownstreamEvaluatorCallbackConfig,
    GPUMemoryMonitorCallback,
    LMEvaluatorCallbackConfig,
    WandBCallback,
)
from olmo_core.train.callbacks.callback import Callback
from olmo_core.train.callbacks.evaluator_callback import DownstreamEvaluator, EvaluatorCallback
from olmo_core.train.train_module import (
    TransformerActivationCheckpointingConfig,
    TransformerActivationCheckpointingMode,
    TransformerDataParallelConfig,
    TransformerTrainModuleConfig,
)
from olmo_core.train.train_module.transformer.config import TransformerMetricsConfig
from olmo_core.utils import seed_all, mark_dynamic


# Set up logging
log = logging.getLogger(__name__)


_XGRAM_TARGET_SET = {"h", "q", "k", "v", "o"}
_QKVO_TARGET_SET = {"q", "k", "v", "o"}
_DEFAULT_DOWNSTREAM_EVAL_TASKS = [
    "mmlu_stem_mc_5shot",
    "mmlu_humanities_mc_5shot",
    "mmlu_social_sciences_mc_5shot",
    "mmlu_other_mc_5shot",
    "arc_easy",
    "boolq",
    "commonsense_qa",
    "hellaswag",
    "openbook_qa",
    "piqa",
    "sciq",
    "social_iqa",
    "winogrande",
    "arc_challenge_test_rc_5shot",
    "csqa_val_rc_5shot",
]


def _canonicalize_injection_mode(raw_mode: Optional[str]) -> str:
    mode = (raw_mode or "None").strip()
    if mode in {"None", "X-gram", "Engram", "Retoken", "Mort"}:
        return mode

    raise ValueError(
        f"Unsupported INJECTION_VERSION='{mode}'. Valid values: None, X-gram, Engram, Retoken, Mort."
    )


def _parse_bool_env(env_name: str, default: str = "0") -> bool:
    return os.environ.get(env_name, default).strip().lower() in {"1", "true", "yes", "on"}


def _parse_optional_int_env(env_name: str) -> Optional[int]:
    raw = os.environ.get(env_name)
    if raw is None or not raw.strip():
        return None
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"Could not parse {env_name}='{raw}' as an integer") from exc


def _parse_optional_float_env(env_name: str) -> Optional[float]:
    raw = os.environ.get(env_name)
    if raw is None or not raw.strip():
        return None
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"Could not parse {env_name}='{raw}' as a float") from exc


def _coerce_bool(value: Any, *, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        if value in (0, 1):
            return bool(value)
        raise ValueError(f"{field_name} must be a boolean or 0/1, got {value}")
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"{field_name} must be a boolean, got {value!r}")


def _coerce_positive_int(value: Any, *, field_name: str) -> int:
    try:
        int_value = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer, got {value!r}") from exc
    if int_value <= 0:
        raise ValueError(f"{field_name} must be > 0, got {int_value}")
    return int_value


def _parse_layer_list_env(env_name: str) -> Optional[List[int]]:
    raw = os.environ.get(env_name, "")
    if not raw:
        return None
    try:
        return [int(layer.strip()) for layer in raw.split(",") if layer.strip()]
    except ValueError as exc:
        raise ValueError(f"Could not parse {env_name}='{raw}': {exc}") from exc


def _parse_positive_int_list_env(env_name: str) -> Optional[List[int]]:
    raw = os.environ.get(env_name, "")
    if not raw:
        return None
    values: List[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            value = int(token)
        except ValueError as exc:
            raise ValueError(f"Could not parse {env_name}='{raw}': {exc}") from exc
        if value <= 0:
            raise ValueError(f"{env_name} only allows positive integers, got {value}")
        values.append(value)
    return values or None


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
        print(
            f"Warning: invalid {env_name} entries: {','.join(invalid)} "
            f"(valid: {','.join(sorted(valid))})"
        )
    if targets:
        return targets
    return list(default) if default is not None else []


def _resolve_injection_target_state(
    injection_version: str,
    *,
    injection_targets_raw: Optional[str],
) -> Tuple[List[str], List[str], bool]:
    if injection_version == "X-gram":
        targets = _parse_ordered_injection_targets(
            injection_targets_raw,
            valid=_XGRAM_TARGET_SET,
            default=["h"],
            env_name="INJECTION_TARGETS",
        )
        return targets, [target for target in targets if target in _QKVO_TARGET_SET], "h" in targets
    return [], [], False


def _resolve_injection_target_state_from_config(
    injection_config: TransformerEmbeddingInjectionConfig,
) -> Tuple[List[str], List[str], bool]:
    if (injection_config.mode or "None") != "X-gram":
        return [], [], False
    targets = list(injection_config.targets or ["h"])
    return targets, [target for target in targets if target in _QKVO_TARGET_SET], "h" in targets


def _build_embedding_injection_config_from_env(
    *,
    warmup_tokens: Optional[int],
    global_batch_size_samples: int,
    seq_len: int,
    streaming_tokenizer_model: str,
) -> Optional[TransformerEmbeddingInjectionConfig]:
    injection_version = _canonicalize_injection_mode(os.environ.get("INJECTION_VERSION", "None"))

    embedding_injection_h_layers = _parse_layer_list_env("INJECTION_H_LAYERS")
    embedding_injection_qk_layers = _parse_layer_list_env("INJECTION_QK_LAYERS")
    embedding_injection_q_layers = _parse_layer_list_env("INJECTION_Q_LAYERS")
    embedding_injection_k_layers = _parse_layer_list_env("INJECTION_K_LAYERS")
    embedding_injection_v_layers = _parse_layer_list_env("INJECTION_V_LAYERS")
    embedding_injection_o_layers = _parse_layer_list_env("INJECTION_O_LAYERS")

    qk_sharing_enabled = _parse_bool_env("INJECTION_QK_SHARING", "0")
    if qk_sharing_enabled and (
        embedding_injection_q_layers is not None or embedding_injection_k_layers is not None
    ):
        raise ValueError(
            "INJECTION_QK_SHARING=1 does not allow INJECTION_Q_LAYERS or INJECTION_K_LAYERS. "
            "Use INJECTION_QK_LAYERS instead."
        )

    embedding_injection_layers: List[int] = []
    for layer_list in (
        embedding_injection_h_layers,
        embedding_injection_qk_layers,
        embedding_injection_q_layers,
        embedding_injection_k_layers,
        embedding_injection_v_layers,
        embedding_injection_o_layers,
    ):
        if layer_list:
            embedding_injection_layers.extend(layer_list)

    if not embedding_injection_layers:
        return None

    injection_targets, _, _ = _resolve_injection_target_state(
        injection_version,
        injection_targets_raw=os.environ.get("INJECTION_TARGETS"),
    )

    lambda_warmup_enabled = injection_version == "X-gram" and _parse_bool_env(
        "INJECTION_LAMBDA_WARMUP_ENABLE", "1"
    )
    lambda_warmup_steps = _parse_optional_int_env("INJECTION_LAMBDA_WARMUP_STEPS")
    if lambda_warmup_steps is None or lambda_warmup_steps <= 0:
        if (
            warmup_tokens is not None
            and warmup_tokens > 0
            and global_batch_size_samples > 0
            and seq_len > 0
        ):
            tokens_per_step = global_batch_size_samples * seq_len
            lambda_warmup_steps = max(1, math.ceil(warmup_tokens / tokens_per_step))
        else:
            lambda_warmup_steps = 0
    lambda_warmup_scale = _parse_optional_float_env("INJECTION_LAMBDA_WARMUP_SCALE")
    if lambda_warmup_scale is not None and lambda_warmup_scale > 0:
        lambda_warmup_steps = max(0, int(math.ceil(lambda_warmup_steps * lambda_warmup_scale)))
    if injection_version == "X-gram" and lambda_warmup_enabled and lambda_warmup_steps == 0:
        raise ValueError(
            "INJECTION_LAMBDA_WARMUP_ENABLE=1 but no valid warmup steps inferred; "
            "please set INJECTION_LAMBDA_WARMUP_STEPS or WARMUP_TOKENS."
        )

    try:
        engram_heads = int(os.environ.get("ENGRAM_ONEGRAM_HEADS", "4"))
    except ValueError as exc:
        raise ValueError("ENGRAM_ONEGRAM_HEADS must be an integer") from exc
    engram_target_buckets_raw = os.environ.get("ENGRAM_ONEGRAM_TARGET_BUCKETS", "75968")
    engram_target_buckets: Optional[int] = None
    if engram_target_buckets_raw is not None:
        try:
            engram_target_buckets = int(engram_target_buckets_raw)
        except ValueError as exc:
            raise ValueError("ENGRAM_ONEGRAM_TARGET_BUCKETS must be an integer") from exc
    try:
        engram_reduction = float(os.environ.get("ENGRAM_ONEGRAM_REDUCTION", "0.5"))
    except ValueError as exc:
        raise ValueError("ENGRAM_ONEGRAM_REDUCTION must be a float") from exc
    try:
        engram_base_seed = int(os.environ.get("ENGRAM_ONEGRAM_SEED", "42"))
    except ValueError as exc:
        raise ValueError("ENGRAM_ONEGRAM_SEED must be an integer") from exc
    try:
        engram_hc_mult = int(os.environ.get("ENGRAM_HC_MULT", "1"))
    except ValueError as exc:
        raise ValueError("ENGRAM_HC_MULT must be an integer") from exc
    try:
        engram_shortconv_kernel = int(os.environ.get("ENGRAM_SHORTCONV_KERNEL", "4"))
    except ValueError:
        engram_shortconv_kernel = 4
    try:
        engram_shortconv_dilation = int(os.environ.get("ENGRAM_SHORTCONV_DILATION", "1"))
    except ValueError:
        engram_shortconv_dilation = 1
    if os.environ.get("ENGRAM_SHORTCONV_KERNELS") not in (None, ""):
        raise ValueError(
            "ENGRAM_SHORTCONV_KERNELS is no longer supported; use ENGRAM_SHORTCONV_KERNEL instead"
        )
    engram_dim_per_ngram = _parse_optional_int_env("ENGRAM_DIM_PER_NGRAM")
    try:
        engram_ngram_heads = int(os.environ.get("ENGRAM_NGRAM_HEADS", "4"))
    except ValueError as exc:
        raise ValueError("ENGRAM_NGRAM_HEADS must be an integer") from exc
    engram_ngram_target_raw = os.environ.get("ENGRAM_NGRAM_TARGET_BUCKETS", "75968")
    engram_ngram_target_buckets: Optional[int] = None
    if engram_ngram_target_raw is not None:
        try:
            engram_ngram_target_buckets = int(engram_ngram_target_raw)
        except ValueError as exc:
            raise ValueError("ENGRAM_NGRAM_TARGET_BUCKETS must be an integer") from exc
    try:
        engram_ngram_seed = int(os.environ.get("ENGRAM_NGRAM_SEED", "137"))
    except ValueError as exc:
        raise ValueError("ENGRAM_NGRAM_SEED must be an integer") from exc

    injection_log_interval = _parse_optional_int_env("INJECTION_LOG_INTERVAL")

    return TransformerEmbeddingInjectionConfig(
        layers=embedding_injection_layers,
        h_layers=embedding_injection_h_layers or [],
        qk_layers=embedding_injection_qk_layers or [],
        q_layers=embedding_injection_q_layers or [],
        k_layers=embedding_injection_k_layers or [],
        v_layers=embedding_injection_v_layers or [],
        o_layers=embedding_injection_o_layers or [],
        mode=injection_version,
        targets=injection_targets,
        qk_sharing=qk_sharing_enabled,
        shortconv_enabled=_parse_bool_env("SHORTCONV_ENABLE", "0"),
        shortconv_kernels=_parse_positive_int_list_env("INJECTION_SC_MULTI_SCALE_KERNELS"),
        hash_enabled=_parse_bool_env("HASH_ENABLE", os.environ.get("V19_HASH_HYBRID", "0")),
        hash_token_map_path=os.environ.get("HASH_TOKEN_MAP_PATH"),
        lambda_init=float(os.environ.get("INJECTION_INIT_LAMBDA", "1.0")),
        lambda_warmup_enabled=lambda_warmup_enabled,
        lambda_warmup_steps=lambda_warmup_steps,
        lambda_warmup_scale=lambda_warmup_scale,
        log_interval=0 if injection_log_interval is None else injection_log_interval,
        depth_scale_disabled=_parse_bool_env("INJECTION_DEPTH_SCALE_DISABLE", "0"),
        engram_tokenizer_id=os.environ.get("ENGRAM_TOKENIZER_ID", streaming_tokenizer_model),
        engram_cache_path=os.environ.get("ENGRAM_LOOKUP_CACHE"),
        engram_heads=engram_heads,
        engram_target_buckets=engram_target_buckets,
        engram_reduction=engram_reduction,
        engram_base_seed=engram_base_seed,
        engram_hc_mult=engram_hc_mult,
        engram_shortconv_enabled=_parse_bool_env("ENGRAM_SHORTCONV_ENABLE", "1"),
        engram_shortconv_kernel=engram_shortconv_kernel,
        engram_shortconv_dilation=engram_shortconv_dilation,
        engram_shortconv_activation=_parse_bool_env("ENGRAM_SHORTCONV_ACTIVATION", "1"),
        engram_mode=os.environ.get("ENGRAM_MODE", "1gram"),
        engram_dim_per_ngram=engram_dim_per_ngram,
        engram_ngram_heads=engram_ngram_heads,
        engram_ngram_target_buckets=engram_ngram_target_buckets,
        engram_ngram_seed=engram_ngram_seed,
        mort_top_k=_parse_optional_int_env("MORT_TOP_K"),
    )


def _parse_targets_value(raw: Any) -> Optional[List[str]]:
    if raw is None:
        return None
    if isinstance(raw, str):
        tokens = [token.strip().lower() for token in raw.split(",") if token.strip()]
    elif isinstance(raw, (list, tuple)):
        tokens = [str(token).strip().lower() for token in raw if str(token).strip()]
    else:
        raise ValueError(f"Unsupported targets value type: {type(raw).__name__}")
    deduped: List[str] = []
    for token in tokens:
        if token not in deduped:
            deduped.append(token)
    return deduped or None


def _infer_xgram_targets_from_yaml(cfg: Dict[str, Any]) -> Optional[List[str]]:
    targets: List[str] = []
    if cfg.get("h_layers"):
        targets.append("h")
    if cfg.get("qk_layers"):
        targets.extend(["q", "k"])
    if cfg.get("q_layers"):
        targets.append("q")
    if cfg.get("k_layers"):
        targets.append("k")
    if cfg.get("v_layers"):
        targets.append("v")
    if cfg.get("o_layers"):
        targets.append("o")
    deduped: List[str] = []
    for token in targets:
        if token not in deduped:
            deduped.append(token)
    return deduped or None


def _apply_embedding_injection_yaml_env_overrides(normalized: Dict[str, Any]) -> None:
    env_overrides = {
        "engram_tokenizer_id": os.environ.get("ENGRAM_TOKENIZER_ID"),
        "engram_cache_path": os.environ.get("ENGRAM_LOOKUP_CACHE"),
    }

    for key, value in env_overrides.items():
        if value is None or value == "":
            continue
        normalized[key] = value
        log.info("Overriding embedding_injection.%s from environment", key)


def _build_embedding_injection_config_from_yaml(
    yaml_path: str,
    *,
    warmup_tokens: Optional[int],
    global_batch_size_samples: int,
    seq_len: int,
) -> Optional[TransformerEmbeddingInjectionConfig]:
    cfg = OmegaConf.load(yaml_path)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(cfg_dict, dict):
        raise ValueError("Embedding injection YAML must deserialize to a mapping")

    injection_cfg = cfg_dict.get("embedding_injection", cfg_dict)
    if injection_cfg is None:
        return None
    if not isinstance(injection_cfg, dict):
        raise ValueError("embedding_injection section must be a mapping")

    mode = _canonicalize_injection_mode(injection_cfg.get("mode"))

    normalized = dict(injection_cfg)
    normalized["mode"] = mode
    normalized["targets"] = _parse_targets_value(normalized.get("targets"))
    _apply_embedding_injection_yaml_env_overrides(normalized)

    if mode == "X-gram" and normalized["targets"] is None:
        normalized["targets"] = _infer_xgram_targets_from_yaml(normalized)

    if "layers" not in normalized or normalized.get("layers") in (None, []):
        combined_layers: List[int] = []
        for key in ("h_layers", "qk_layers", "q_layers", "k_layers", "v_layers", "o_layers"):
            values = normalized.get(key)
            if values:
                combined_layers.extend(list(values))
        normalized["layers"] = combined_layers

    if not normalized.get("layers"):
        raise ValueError("embedding_injection YAML must specify at least one layer list")

    if "lambda_warmup_enabled" in normalized:
        lambda_warmup_enabled = bool(normalized.get("lambda_warmup_enabled"))
    else:
        lambda_warmup_enabled = mode == "X-gram"
    normalized["lambda_warmup_enabled"] = lambda_warmup_enabled
    lambda_warmup_steps = normalized.get("lambda_warmup_steps")
    if lambda_warmup_steps is None or int(lambda_warmup_steps) <= 0:
        if (
            mode == "X-gram"
            and lambda_warmup_enabled
            and warmup_tokens is not None
            and warmup_tokens > 0
            and global_batch_size_samples > 0
            and seq_len > 0
        ):
            tokens_per_step = global_batch_size_samples * seq_len
            normalized["lambda_warmup_steps"] = max(1, math.ceil(warmup_tokens / tokens_per_step))

    return TransformerEmbeddingInjectionConfig.from_dict(normalized)


def _load_public_yaml_config(yaml_path: str) -> Dict[str, Any]:
    cfg = OmegaConf.load(yaml_path)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(cfg_dict, dict):
        raise ValueError("Public YAML config must deserialize to a mapping")
    return cfg_dict


@dataclass
class ResolvedLaunch:
    run_name: str
    save_root: str
    save_folder: str
    streaming_data_path: List[str]
    streaming_tokenizer_model: str
    streaming_ckpt_path: str
    launcher_mode: str
    run_name_source: str
    save_root_source: str
    save_folder_source: str
    streaming_data_path_source: str
    streaming_tokenizer_model_source: str
    streaming_ckpt_path_source: str
    streaming_local_cache_path: Optional[str] = None
    staged_ckpt_source: Optional[str] = None

    def as_metadata(self) -> Dict[str, Any]:
        return {
            "run_name": self.run_name,
            "run_name_source": self.run_name_source,
            "save_root": self.save_root,
            "save_root_source": self.save_root_source,
            "save_folder": self.save_folder,
            "save_folder_source": self.save_folder_source,
            "streaming_data_path": list(self.streaming_data_path),
            "streaming_data_path_source": self.streaming_data_path_source,
            "streaming_tokenizer_model": self.streaming_tokenizer_model,
            "streaming_tokenizer_model_source": self.streaming_tokenizer_model_source,
            "streaming_ckpt_path": self.streaming_ckpt_path,
            "streaming_ckpt_path_source": self.streaming_ckpt_path_source,
            "launcher_mode": self.launcher_mode,
            "streaming_local_cache_path": self.streaming_local_cache_path,
            "staged_ckpt_source": self.staged_ckpt_source,
        }


def _resolve_streaming_local_path() -> str:
    streaming_cache_base = os.environ.get("STREAMING_CACHE_BASE", "/tmp/dclm-baseline-0.25b-l10-streaming-v2")
    job_id_tag = os.environ.get("JOB_ID") or os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOBID")
    host_tag = socket.gethostname().split(".", 1)[0] or "unknown-host"
    launch_time_tag = os.environ.get("LAUNCH_TIME_TAG")
    if not launch_time_tag:
        launch_time_tag = datetime.now().strftime("%Y%m%d-%H%M%S")
    if job_id_tag:
        streaming_local_path = f"{streaming_cache_base}-{job_id_tag}-{host_tag}"
    else:
        streaming_local_path = f"{streaming_cache_base}-{launch_time_tag}-{host_tag}"
    os.makedirs(streaming_local_path, exist_ok=True)
    return streaming_local_path


def _replace_streaming_local_cache_slot(paths: List[Any], local_path: str) -> List[str]:
    resolved = [str(x) for x in paths]
    if len(resolved) >= 2:
        resolved[1] = local_path
    return resolved


def _resolve_launcher_streaming_data_path(
    *,
    explicit_streaming_data_path: Optional[List[str]],
    data_yaml: Dict[str, Any],
    launcher_mode: str,
) -> Tuple[List[str], str, Optional[str]]:
    if explicit_streaming_data_path is not None:
        return (
            [str(x) for x in explicit_streaming_data_path],
            "cli --streaming-data-path",
            None,
        )

    if launcher_mode == "shell_compatible":
        streaming_local_path = _resolve_streaming_local_path()
        raw_streaming_data_path = data_yaml.get("streaming_data_path")
        if raw_streaming_data_path is not None and not isinstance(raw_streaming_data_path, list):
            raise ValueError("data.streaming_data_path must be a list")
        streaming_data_path = list(raw_streaming_data_path or [])
        if streaming_data_path:
            return (
                _replace_streaming_local_cache_slot(streaming_data_path, streaming_local_path),
                "launcher-mode shell_compatible + yaml data.streaming_data_path",
                streaming_local_path,
            )
        return (
            [
                "1",
                streaming_local_path,
                "/tmp/dclm-baseline-1.0-streaming",
            ],
            "launcher-mode shell_compatible fallback",
            streaming_local_path,
        )

    streaming_data_path = data_yaml.get("streaming_data_path")
    if streaming_data_path is None:
        raise ValueError(
            "streaming_data_path must be provided either via CLI or YAML data.streaming_data_path"
        )
    if not isinstance(streaming_data_path, list):
        raise ValueError("data.streaming_data_path must be a list")
    return ([str(x) for x in streaming_data_path], "yaml data.streaming_data_path", None)


def _resolve_launch_inputs(
    *,
    run_name: Optional[str],
    save_root: Optional[str],
    streaming_data_path: Optional[List[str]],
    streaming_tokenizer_model: Optional[str],
    streaming_ckpt_path: Optional[str],
    run_yaml: Dict[str, Any],
    data_yaml: Dict[str, Any],
    launcher_mode: str,
) -> ResolvedLaunch:
    if run_name is not None:
        run_name_value = str(run_name)
        run_name_source = "cli --run-name"
    elif run_yaml.get("run_name") is not None:
        run_name_value = str(run_yaml["run_name"])
        run_name_source = "yaml run.run_name"
    elif run_yaml.get("experiment_name") is not None:
        run_name_value = str(run_yaml["experiment_name"])
        run_name_source = "yaml run.experiment_name"
    else:
        raise ValueError(
            "run_name must be provided either via CLI or YAML run.run_name / run.experiment_name"
        )

    if save_root is not None:
        save_root_value = str(save_root)
        save_root_source = "cli --save-root"
    else:
        save_root_yaml = run_yaml.get("save_root")
        if save_root_yaml is None:
            raise ValueError("save_root must be provided either via CLI or YAML run.save_root")
        save_root_value = str(save_root_yaml)
        save_root_source = "yaml run.save_root"
    save_folder_value = os.path.join(save_root_value, run_name_value)
    save_folder_source = "save_root + run_name"

    (
        streaming_data_path_value,
        streaming_data_path_source,
        streaming_local_cache_path,
    ) = _resolve_launcher_streaming_data_path(
        explicit_streaming_data_path=streaming_data_path,
        data_yaml=data_yaml,
        launcher_mode=launcher_mode,
    )

    if streaming_tokenizer_model is not None:
        streaming_tokenizer_model_value = str(streaming_tokenizer_model)
        streaming_tokenizer_model_source = "cli --streaming-tokenizer-model"
    elif data_yaml.get("streaming_tokenizer_model") is not None:
        streaming_tokenizer_model_value = str(data_yaml["streaming_tokenizer_model"])
        streaming_tokenizer_model_source = "yaml data.streaming_tokenizer_model"
    elif data_yaml.get("tokenizer_model") is not None:
        streaming_tokenizer_model_value = str(data_yaml["tokenizer_model"])
        streaming_tokenizer_model_source = "yaml data.tokenizer_model"
    else:
        raise ValueError(
            "streaming_tokenizer_model must be provided either via CLI or YAML data.streaming_tokenizer_model"
        )

    if streaming_ckpt_path is not None:
        streaming_ckpt_path_value = str(streaming_ckpt_path)
        streaming_ckpt_path_source = "cli --streaming-ckpt-path"
    elif data_yaml.get("streaming_ckpt_path") is not None:
        streaming_ckpt_path_value = str(data_yaml["streaming_ckpt_path"])
        streaming_ckpt_path_source = "yaml data.streaming_ckpt_path"
    else:
        raise ValueError(
            "streaming_ckpt_path must be provided either via CLI or YAML data.streaming_ckpt_path"
        )

    return ResolvedLaunch(
        run_name=run_name_value,
        save_root=save_root_value,
        save_folder=save_folder_value,
        streaming_data_path=streaming_data_path_value,
        streaming_tokenizer_model=streaming_tokenizer_model_value,
        streaming_ckpt_path=streaming_ckpt_path_value,
        launcher_mode=launcher_mode,
        run_name_source=run_name_source,
        save_root_source=save_root_source,
        save_folder_source=save_folder_source,
        streaming_data_path_source=streaming_data_path_source,
        streaming_tokenizer_model_source=streaming_tokenizer_model_source,
        streaming_ckpt_path_source=streaming_ckpt_path_source,
        streaming_local_cache_path=streaming_local_cache_path,
    )


def _log_resolved_launch_metadata(resolved: ResolvedLaunch) -> None:
    log.info("Resolved launch metadata:")
    log.info("- run_name=%s (%s)", resolved.run_name, resolved.run_name_source)
    log.info("- save_root=%s (%s)", resolved.save_root, resolved.save_root_source)
    log.info("- save_folder=%s (%s)", resolved.save_folder, resolved.save_folder_source)
    log.info(
        "- streaming_data_path=%s (%s)",
        resolved.streaming_data_path,
        resolved.streaming_data_path_source,
    )
    log.info(
        "- streaming_tokenizer_model=%s (%s)",
        resolved.streaming_tokenizer_model,
        resolved.streaming_tokenizer_model_source,
    )
    log.info(
        "- streaming_ckpt_path=%s (%s)",
        resolved.streaming_ckpt_path,
        resolved.streaming_ckpt_path_source,
    )
    log.info("- launcher_mode=%s", resolved.launcher_mode)
    if resolved.streaming_local_cache_path:
        log.info("- streaming_local_cache_path=%s", resolved.streaming_local_cache_path)
    if resolved.staged_ckpt_source:
        log.info("- staged_ckpt_source=%s", resolved.staged_ckpt_source)


def _resolve_default_save_tokens(train_tokens: int, seq_len: int, global_batch_size_samples: int) -> int:
    tokens_per_step = max(1, seq_len * global_batch_size_samples)
    save_fraction = float(os.environ.get("OLMO_SAVE_TOKENS_FRACTION", "0.25"))
    if save_fraction <= 0:
        raise ValueError(f"OLMO_SAVE_TOKENS_FRACTION must be > 0, got {save_fraction}")
    derived_save_tokens = int(math.ceil(train_tokens * save_fraction))
    return max(tokens_per_step, derived_save_tokens)


def _resolve_ac_config(ac_mode: Optional[str]) -> Optional[TransformerActivationCheckpointingConfig]:
    mode = (ac_mode or "").strip().lower()
    if mode in {"", "none", "off", "disabled"}:
        return None
    try:
        mode_enum = TransformerActivationCheckpointingMode(mode)
    except ValueError as exc:
        valid_modes = ", ".join(m.value for m in TransformerActivationCheckpointingMode)
        raise ValueError(f"Unsupported activation checkpointing mode '{ac_mode}'. Valid values: {valid_modes}") from exc
    return TransformerActivationCheckpointingConfig(mode=mode_enum)


def _derive_default_init_std(hidden_size: int) -> float:
    return float(f"{math.sqrt(2.0 / (5.0 * hidden_size)):.3f}")


def _format_count(count: int) -> str:
    """Format large numbers with K/M/B suffixes."""
    if count >= 1_000_000_000:
        return f"{count / 1_000_000_000:.1f}B"
    elif count >= 1_000_000:
        return f"{count / 1_000_000:.1f}M"
    elif count >= 1_000:
        return f"{count / 1_000:.1f}K"
    else:
        return str(count)


def _module_param_stats(module: torch.nn.Module, name: str) -> Dict[str, Any]:
    """Compute parameter statistics for a module."""
    params = list(module.parameters())
    if not params:
        return {"name": name, "params": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    
    param_count = sum(p.numel() for p in params)
    all_params = torch.cat([p.flatten() for p in params])
    
    return {
        "name": name,
        "params": param_count,
        "mean": all_params.mean().item(),
        "std": all_params.std().item(),
        "min": all_params.min().item(),
        "max": all_params.max().item(),
    }

def log_model_diagnostics(config: Config) -> None:
    """Log detailed model diagnostics including parameter counts, FLOPs, and layer statistics."""
    if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
        return

    log.info("Collecting model diagnostics...")

    # Build the model for statistics on CPU to avoid GPU memory use.
    stats_model = config.model.build(init_device="meta")
    with torch.no_grad():
        stats_model.init_weights(
            max_seq_len=config.train_module.max_sequence_length,
            max_local_microbatch_size=config.train_module.rank_microbatch_size,
            device=torch.device("cpu"),
        )
    stats_model = stats_model.cpu()

    # Basic model parameters
    total_params = config.model.num_params
    active_params = config.model.num_active_params
    non_embedding_params = config.model.num_non_embedding_params
    active_non_embedding = config.model.num_active_non_embedding_params

    seq_len = config.train_module.max_sequence_length
    flops_per_token = config.model.num_flops_per_token(seq_len)
    tokens_per_rank_step = config.train_module.rank_microbatch_size
    tokens_per_global_step = config.data_loader.ub_global_batch_size * seq_len
    flops_per_rank_step = flops_per_token * tokens_per_rank_step
    flops_per_global_step = flops_per_token * tokens_per_global_step

    d_model = config.model.d_model
    n_layers = config.model.n_layers
    n_heads = config.model.block.attention.n_heads
    head_dim = d_model // n_heads if n_heads else 0

    log.info(
        "Model configuration:\n"
        f"- width (d_model): {d_model:,d}\n"
        f"- depth (layers): {n_layers:,d}\n"
        f"- attention heads: {n_heads:,d}\n"
        f"- per-head dimension (d/n): {head_dim:,d}\n"
        f"- vocab size: {config.model.vocab_size:,d}"
    )
    
    log.info(
        "Parameter counts:\n"
        f"- total params: {_format_count(total_params)}\n"
        f"- active params: {_format_count(active_params)}\n"
        f"- non-embedding params: {_format_count(non_embedding_params)}\n"
        f"- active non-embedding params: {_format_count(active_non_embedding)}"
    )
    
    log.info(
        "Approximate FLOPs:\n"
        f"- per token: {_format_count(flops_per_token)}\n"
        f"- per rank step: {_format_count(flops_per_rank_step)}\n"
        f"- per global step: {_format_count(flops_per_global_step)}"
    )

    log.info("Parameter statistics by component:")

    if hasattr(stats_model, 'embeddings'):
        emb_stats = _module_param_stats(stats_model.embeddings, "embeddings")
        log.info(f"- {emb_stats['name']}: {_format_count(emb_stats['params'])} params, "
                f"mean={emb_stats['mean']:.4f}, std={emb_stats['std']:.4f}, "
                f"min={emb_stats['min']:.4f}, max={emb_stats['max']:.4f}")

    if hasattr(stats_model, 'blocks'):
        for i, block in enumerate(stats_model.blocks.values()):
            block_stats = _module_param_stats(block, f"block_{i:02d}")
            log.info(f"- {block_stats['name']}: {_format_count(block_stats['params'])} params, "
                    f"mean={block_stats['mean']:.4f}, std={block_stats['std']:.4f}, "
                    f"min={block_stats['min']:.4f}, max={block_stats['max']:.4f}")

    if hasattr(stats_model, 'lm_head'):
        lm_head_stats = _module_param_stats(stats_model.lm_head, "lm_head")
        log.info(f"- {lm_head_stats['name']}: {_format_count(lm_head_stats['params'])} params, "
                f"mean={lm_head_stats['mean']:.4f}, std={lm_head_stats['std']:.4f}, "
                f"min={lm_head_stats['min']:.4f}, max={lm_head_stats['max']:.4f}")

    log.info("Model diagnostics collection complete.")


@dataclass
class ExperimentConfig(Config):
    model: TransformerConfig
    data_loader: UBDataLoaderConfig
    train_module: TransformerTrainModuleConfig
    trainer: TrainerConfig
    init_seed: int = 42


class TensorBoardCallback(Callback):
    enabled: bool = True
    log_dir: Optional[str] = None

    def __init__(self, *, log_dir: Optional[str] = None, enabled: bool = True):
        super().__init__()
        self.enabled = enabled
        self.log_dir = log_dir
        self._writer = None

    def pre_train(self):
        if not self.enabled:
            return
        try:
            from torch.utils.tensorboard import SummaryWriter  # type: ignore
        except Exception:
            self.enabled = False
            return
        log_dir = self.log_dir or os.path.join(self.trainer.save_folder, "tensorboard")
        os.makedirs(log_dir, exist_ok=True)
        self._writer = SummaryWriter(log_dir=log_dir)

    def log_metrics(self, step: int, metrics):
        if not self.enabled or self._writer is None:
            return
        for k, v in metrics.items():
            try:
                self._writer.add_scalar(k, float(v), step)
            except Exception:
                continue

    def post_train(self):
        if self._writer is not None:
            try:
                self._writer.flush()
                self._writer.close()
            except Exception:
                pass


def _to_dtype(name: str) -> DType:
    name = name.lower()
    if name == "float32":
        return DType.float32
    if name == "bfloat16":
        return DType.bfloat16
    if name == "float16":
        return DType.float16
    raise ValueError(f"Unsupported dtype: {name}")


def _build_tokenizer_config(identifier: Optional[str]) -> TokenizerConfig:
    """Build TokenizerConfig from a local HF model folder or HF hub id."""
    if not identifier:
        return TokenizerConfig.dolma2()
    if os.path.isdir(identifier):
        cfg_path = os.path.join(identifier, "config.json")
        if not os.path.isfile(cfg_path):
            raise RuntimeError(f"tokenizer-model directory missing config.json: {identifier}")
        import json
        with open(cfg_path, "r") as f:
            cfg = json.load(f)
        return TokenizerConfig(
            vocab_size=cfg["vocab_size"],
            eos_token_id=cfg["eos_token_id"],
            pad_token_id=cfg.get("pad_token_id", cfg["eos_token_id"]),
            bos_token_id=cfg.get("bos_token_id"),
            identifier=os.path.join(identifier, "tokenizer.json"),  
        )
    return TokenizerConfig.from_hf(identifier)




def build_config(
    *,
    run_name: str,
    streaming_data_path: List[str],
    streaming_tokenizer_model: str,
    streaming_ckpt_path: str,
    save_root: str,
    seq_len: int,
    micro_batch_size: int,
    global_batch_size_samples: int,
    train_tokens: int,
    warmup_fraction: float,
    warmup_tokens: Optional[int],
    save_tokens: int,
    lr: float,
    min_lr: float,
    weight_decay: float,
    beta1: float,
    beta2: float,
    num_layers: int,
    hidden_size: int,
    ffn_hidden_size: int,
    num_attn_heads: int,
    num_query_groups: int,
    qk_norm: bool,
    rope_theta: int,
    layer_norm_eps: float,
    dp_type: str,
    param_dtype: DType,
    reduce_dtype: DType,
    grad_clip: float,
    ac_mode: str,
    enable_wandb: bool,
    enable_comet: bool,
    downstream_enabled: bool,
    downstream_eval_interval: int,
    eval_iters: int,
    log_interval: int = 10,
    text_chunk_queue_size: int = 8,
    text_chunk_size: int = 40960,
    prefetch_queue_size: int = 8,
    use_token_column: Optional[str] = None,
    pack_method: str = "native:truncate",
    init_std: Optional[float] = None,
    embedding_injection: Optional[TransformerEmbeddingInjectionConfig] = None,
    eval_data_work_dir: Optional[str] = None,
) -> ExperimentConfig:
    # Tokenizer: use provided local HF folder or hub id.
    tokenizer_config = _build_tokenizer_config(streaming_tokenizer_model)

    # Tokens-based batching in OLMo: specify sizes in tokens, not samples.
    global_batch_size_tokens = global_batch_size_samples * seq_len
    rank_microbatch_size_tokens = micro_batch_size * seq_len

    data_loader_config = UBDataLoaderConfig(
        data_path=streaming_data_path,
        seq_len=seq_len,
        ckpt_path=streaming_ckpt_path,
        ub_global_batch_size=global_batch_size_samples,  
        tokenizer_path=streaming_tokenizer_model,
        text_chunk_queue_size=text_chunk_queue_size,
        text_chunk_size=text_chunk_size,
        prefetch_queue_size=prefetch_queue_size,
        use_token_column=use_token_column,
        pack_method=pack_method,
    )

    # Match exact FFN size via hidden_size_multiplier and multiple-of.
    hidden_size_multiplier = ffn_hidden_size / float(hidden_size)

    if init_std is None:
        init_std = float(os.environ.get("INIT_STD", "0.02"))
        print(f"Using INIT_STD: {init_std} (from environment: {os.environ.get('INIT_STD', 'not set, using default')})")
    else:
        print(f"Using INIT_STD: {init_std:.3f} (derived from hidden_size={hidden_size})")
    
    injection_config = embedding_injection

    model_config = TransformerConfig.llama_like(
        vocab_size=tokenizer_config.padded_vocab_size(),
        d_model=hidden_size,
        n_layers=num_layers,
        n_heads=num_attn_heads,
        n_kv_heads=num_query_groups,
        hidden_size_multiplier=hidden_size_multiplier,
        hidden_size_multiple_of=1,
        qk_norm=qk_norm,
        rope_theta=rope_theta,
        layer_norm_eps=layer_norm_eps,
        feed_forward=FeedForwardConfig(hidden_size=ffn_hidden_size, bias=False),
        init_std=init_std,  
        use_flash=False,
        embedding_injection=injection_config,
    )

    warmup_frac = (warmup_tokens / float(train_tokens)) if (warmup_tokens is not None and train_tokens > 0) else warmup_fraction
    min_lr_effective = min_lr
    scheduler = CosWithWarmup(
        warmup_steps=None,
        warmup_fraction=warmup_frac,
        alpha_f=min_lr_effective / lr,
    )

    compile_flag = True

    group_overrides: List[OptimGroupOverride] = []
    if injection_config is not None:
        injection_version_env = injection_config.mode or "None"
        injection_targets, qkvo_targets, h_target_enabled = _resolve_injection_target_state_from_config(
            injection_config
        )
        qkvo_target_set = set(qkvo_targets)
        qk_sharing_enabled = injection_config.qk_sharing
        qk_shared_active = (
            qk_sharing_enabled
            and bool(injection_config.qk_layers)
            and {"q", "k"}.issubset(qkvo_target_set)
        )
        h_active = h_target_enabled and bool(injection_config.h_layers)
        q_active = "q" in qkvo_target_set and (
            qk_shared_active or bool(injection_config.q_layers) or bool(injection_config.h_layers)
        )
        k_active = "k" in qkvo_target_set and (
            qk_shared_active or bool(injection_config.k_layers) or bool(injection_config.h_layers)
        )
        v_active = "v" in qkvo_target_set and (
            bool(injection_config.v_layers) or bool(injection_config.h_layers)
        )
        o_active = "o" in qkvo_target_set and (
            bool(injection_config.o_layers) or bool(injection_config.h_layers)
        )
        hash_enabled = injection_config.hash_enabled

        if injection_version_env == "Retoken":
            pass
        elif injection_version_env == "Mort":
            pass
        elif injection_version_env == "X-gram" and hash_enabled:
            vocab_size = tokenizer_config.padded_vocab_size()
            d_model = hidden_size
            override_lr = lr * math.sqrt(vocab_size / d_model)
            print(f"Override LR: {override_lr}")
            override_params = []
            if h_active:
                override_params.append("_injection_h_embeddings.*._bucket_embedding.weight")
            if qk_shared_active:
                override_params.append("_injection_qk_embeddings.*._bucket_embedding.weight")
            else:
                if q_active:
                    override_params.append("_injection_q_embeddings.*._bucket_embedding.weight")
                if k_active:
                    override_params.append("_injection_k_embeddings.*._bucket_embedding.weight")
            if v_active:
                override_params.append("_injection_v_embeddings.*._bucket_embedding.weight")
            if o_active:
                override_params.append("_injection_o_embeddings.*._bucket_embedding.weight")
            group_overrides.append(
                OptimGroupOverride(
                    params=override_params,
                    opts=dict(lr=override_lr),
                )
            )

        else:
            vocab_size = tokenizer_config.padded_vocab_size()
            d_model = hidden_size
            override_lr = lr * math.sqrt(vocab_size / d_model)
            print(f"Override LR: {override_lr}")

            override_params = []
            if h_active:
                override_params.append("_injection_h_embeddings.*.weight")
            if qk_shared_active:
                override_params.append("_injection_qk_embeddings.*.weight")
            else:
                if q_active:
                    override_params.append("_injection_q_embeddings.*.weight")
                if k_active:
                    override_params.append("_injection_k_embeddings.*.weight")
            if v_active:
                override_params.append("_injection_v_embeddings.*.weight")
            if o_active:
                override_params.append("_injection_o_embeddings.*.weight")
            elif injection_version_env == "Engram":
                _engram_mode_parts = {
                    s.strip() for s in (injection_config.engram_mode or "1gram").split("+")
                }
                override_params = []
                if "1gram" in _engram_mode_parts:
                    override_params.append("_injection_h_embeddings.*.embedding.weight")
                if any(p != "1gram" for p in _engram_mode_parts):
                    override_params.append("_injection_h_embeddings.*.ngram_embeddings.*.weight")

            group_overrides.append(
                OptimGroupOverride(
                    params=override_params,
                    opts=dict(lr=override_lr),
                )
            )

    if group_overrides:
        print("Configured optimizer group overrides:")
        for idx, override in enumerate(group_overrides, start=1):
            print(f"  [{idx}] params={override.params} opts={override.opts}")
    else:
        print("No optimizer group overrides configured.")

    train_module_config = TransformerTrainModuleConfig(
        rank_microbatch_size=rank_microbatch_size_tokens,
        max_sequence_length=seq_len,
        optim=SkipStepAdamWConfig(
            lr=lr,
            weight_decay=weight_decay,
            betas=(beta1, beta2),
            compile=compile_flag,
            group_overrides=group_overrides if group_overrides else None,
        ),
        scheduler=scheduler,
        compile_model=compile_flag,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.hsdp if dp_type == "hsdp" else DataParallelType.fsdp,
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
        ),
        ac_config=_resolve_ac_config(ac_mode),
        z_loss_multiplier=1e-5,
        max_grad_norm=grad_clip,
        metrics=TransformerMetricsConfig(
            enabled=True,
            log_batch_size=True,
            log_effective_length=True,
            log_loss_per_byte=True,
            log_layer_grad_norm=True,
            log_module_grad_norm=True,
            log_last_activation_norm=True,
            log_embedding_norm=True,
            log_param_mean=True,
            param_mean_keywords=("embedding", "injection", "gate", "scalar"),
            log_moe_metrics=True,
        ),
    )

    save_interval_steps = max(1, save_tokens // global_batch_size_tokens)
    ephemeral_save_interval_candidate = max(1, save_interval_steps // 10)
    ephemeral_save_interval = min(ephemeral_save_interval_candidate, save_interval_steps - 1) if save_interval_steps > 1 else None

    trainer_config = (
        TrainerConfig(
            save_folder=f"{save_root}/{run_name}",
            save_overwrite=True,
            load_strategy=LoadStrategy.never,
            metrics_collect_interval=log_interval,
            cancel_check_interval=10,
            max_duration=Duration.tokens(train_tokens),
        )
        .with_callback("gpu_monitor", GPUMemoryMonitorCallback())
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=save_interval_steps,
                ephemeral_save_interval=ephemeral_save_interval,
                save_async=True,
            ),
        )
        .with_callback(
            "comet",
            CometCallback(
                name=run_name,
                enabled=enable_comet,
                cancel_check_interval=10,
            ),
        )
        .with_callback(
            "wandb",
            WandBCallback(
                name=run_name,
                enabled=enable_wandb or bool(os.environ.get("WANDB_API_KEY")),
                project=os.environ.get("WANDB_PROJECT") or None,
                entity=os.environ.get("WANDB_ENTITY") or None,
            ),
        )
        .with_callback("config_saver", ConfigSaverCallback())
        .with_callback(
            "tensorboard",
            TensorBoardCallback(
                log_dir=None,
                enabled=True,
            ),
        )
        .with_callback(
            "downstream_evaluator",
            FilteredDownstreamEvaluatorCallbackConfig(
                tasks=list(_DEFAULT_DOWNSTREAM_EVAL_TASKS),
                tokenizer=tokenizer_config,
                eval_interval=downstream_eval_interval,
                eval_on_startup=True,
                enabled=downstream_enabled,
            ),
        )
    )
    return ExperimentConfig(
        model=model_config,
        data_loader=data_loader_config,
        train_module=train_module_config,
        trainer=trainer_config,
        init_seed=42,
    )




def main(
    run_name: Optional[str],
    streaming_data_path: Optional[List[str]],
    streaming_tokenizer_model: Optional[str],
    streaming_ckpt_path: Optional[str],
    save_root: Optional[str],
    micro_batch_size: Optional[int],
    streaming_text_chunk_queue_size: Optional[int],
    streaming_text_chunk_size: Optional[int],
    streaming_prefetch_queue_size: Optional[int],
    streaming_use_token_column: Optional[str],
    streaming_pack_method: Optional[str],
    config_path: Optional[str],
    embedding_injection_yaml: Optional[str],
    launcher_mode: str,
    print_resolved_config: bool,
    overrides: List[str],
):
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    yaml_path = config_path or embedding_injection_yaml
    public_yaml_cfg: Dict[str, Any] = {}
    if yaml_path is not None:
        public_yaml_cfg = _load_public_yaml_config(yaml_path)
    run_yaml = public_yaml_cfg.get("run", {})
    if not isinstance(run_yaml, dict):
        raise ValueError("run section in YAML must be a mapping")
    data_yaml = public_yaml_cfg.get("data", {})
    if not isinstance(data_yaml, dict):
        raise ValueError("data section in YAML must be a mapping")
    training_yaml = public_yaml_cfg.get("training", {})
    if not isinstance(training_yaml, dict):
        raise ValueError("training section in YAML must be a mapping")
    model_yaml = public_yaml_cfg.get("model", {})
    if not isinstance(model_yaml, dict):
        raise ValueError("model section in YAML must be a mapping")
    evaluation_yaml = public_yaml_cfg.get("evaluation", {})
    if evaluation_yaml is None:
        evaluation_yaml = {}
    if not isinstance(evaluation_yaml, dict):
        raise ValueError("evaluation section in YAML must be a mapping")
    downstream_yaml = evaluation_yaml.get("downstream", {})
    if downstream_yaml is None:
        downstream_yaml = {}
    if not isinstance(downstream_yaml, dict):
        raise ValueError("evaluation.downstream section in YAML must be a mapping")

    SEQ_LEN = int(training_yaml.get("seq_len", os.environ.get("SEQ_LEN", 8192)))
    micro_batch_value = micro_batch_size if micro_batch_size is not None else training_yaml.get("micro_batch_size")
    if micro_batch_value is None:
        raise ValueError("micro_batch_size must be provided either via CLI or YAML training.micro_batch_size")
    MICRO_BATCH_SIZE = int(micro_batch_value)
    GLOBAL_BATCH_SIZE_SAMPLES = int(
        training_yaml.get("global_batch_size", os.environ.get("GLOBAL_BATCH_SIZE", 1024))
    )

    TRAIN_TOKENS = int(
        training_yaml.get("train_tokens", os.environ.get("TRAIN_TOKENS", 41_400_000_000))
    )
    LR_DECAY_TOKENS = int(os.environ.get("OLMO_LR_DECAY_TOKENS", TRAIN_TOKENS))
    WARMUP_FRACTION = float(training_yaml.get("warmup_fraction", os.environ.get("OLMO_WARMUP_FRACTION", 0.05)))
    if yaml_path is not None:
        WARMUP_TOKENS = int(TRAIN_TOKENS * WARMUP_FRACTION)
    else:
        WARMUP_TOKENS = int(os.environ["WARMUP_TOKENS"]) if os.environ.get("WARMUP_TOKENS") else None
    save_tokens_value = training_yaml.get("save_tokens")
    if save_tokens_value is not None:
        SAVE_TOKENS = int(save_tokens_value)
    else:
        explicit_save_tokens = os.environ.get("SAVE_TOKENS", os.environ.get("OLMO_SAVE_TOKENS"))
        if explicit_save_tokens is not None:
            SAVE_TOKENS = int(explicit_save_tokens)
        else:
            SAVE_TOKENS = _resolve_default_save_tokens(
                TRAIN_TOKENS,
                seq_len=SEQ_LEN,
                global_batch_size_samples=GLOBAL_BATCH_SIZE_SAMPLES,
            )

    LR = float(training_yaml.get("lr", os.environ.get("LR", 1e-3)))
    min_lr_value = training_yaml.get("min_lr")
    if min_lr_value is not None:
        MIN_LR = float(min_lr_value)
    elif yaml_path is not None:
        MIN_LR = LR * 0.1
    else:
        MIN_LR = float(os.environ.get("MIN_LR", 1e-4))
    WEIGHT_DECAY = float(os.environ.get("OLMO_WEIGHT_DECAY", 0.1))
    BETA1 = float(os.environ.get("OLMO_BETA1", 0.9))
    BETA2 = float(os.environ.get("OLMO_BETA2", 0.95))

    NUM_LAYERS = int(model_yaml.get("num_layers", os.environ.get("NUM_LAYERS", 10)))
    HIDDEN_SIZE = int(model_yaml.get("hidden_size", os.environ.get("HIDDEN_SIZE", 1536)))
    FFN_HIDDEN_SIZE = int(model_yaml.get("ffn_hidden_size", os.environ.get("FFN_HIDDEN_SIZE", 4096)))
    NUM_ATTN_HEADS = int(model_yaml.get("num_attn_heads", os.environ.get("NUM_ATTN_HEADS", 12)))
    NUM_QUERY_GROUPS = int(model_yaml.get("num_query_groups", os.environ.get("NUM_QUERY_GROUPS", 6)))
    QK_NORM = True
    ROPE_THETA = 500_000
    LAYER_NORM_EPS = 1e-8

    DP_TYPE = os.environ.get("OLMO_DP_TYPE", "hsdp")
    PARAM_DTYPE = _to_dtype("bfloat16")
    REDUCE_DTYPE = _to_dtype("float32")
    GRAD_CLIP = float(os.environ.get("GRAD_CLIP", 1.0))
    AC_MODE = str(training_yaml.get("ac_mode", os.environ.get("OLMO_AC_MODE", "full")))
    LOG_INTERVAL = int(training_yaml.get("log_interval", os.environ.get("OLMO_LOG_INTERVAL", 10)))
    TEXT_CHUNK_QUEUE_SIZE = (
        streaming_text_chunk_queue_size
        if streaming_text_chunk_queue_size is not None
        else int(data_yaml.get("text_chunk_queue_size", os.environ.get("TEXT_CHUNK_QUEUE_SIZE", 8)))
    )
    TEXT_CHUNK_SIZE = (
        streaming_text_chunk_size
        if streaming_text_chunk_size is not None
        else int(data_yaml.get("text_chunk_size", os.environ.get("TEXT_CHUNK_SIZE", 40960)))
    )
    PREFETCH_QUEUE_SIZE = (
        streaming_prefetch_queue_size
        if streaming_prefetch_queue_size is not None
        else int(data_yaml.get("prefetch_queue_size", os.environ.get("PREFETCH_QUEUE_SIZE", 8)))
    )
    USE_TOKEN_COLUMN = (
        streaming_use_token_column
        if streaming_use_token_column is not None
        else data_yaml.get("use_token_column", os.environ.get("STREAMING_USE_TOKEN_COLUMN"))
    )
    PACK_METHOD = (
        streaming_pack_method
        if streaming_pack_method is not None
        else data_yaml.get("pack_method", os.environ.get("STREAMING_PACK_METHOD", "native:truncate"))
    )
    resolved = _resolve_launch_inputs(
        run_name=run_name,
        save_root=save_root,
        streaming_data_path=streaming_data_path,
        streaming_tokenizer_model=streaming_tokenizer_model,
        streaming_ckpt_path=streaming_ckpt_path,
        run_yaml=run_yaml,
        data_yaml=data_yaml,
        launcher_mode=launcher_mode,
    )

    downstream_enabled_value = downstream_yaml.get("enabled", False)
    DOWNSTREAM_ENABLED = _coerce_bool(
        downstream_enabled_value,
        field_name="evaluation.downstream.enabled",
    )
    downstream_eval_interval_value = downstream_yaml.get("eval_interval")
    if downstream_eval_interval_value is None:
        downstream_eval_interval_value = 400
    DOWNSTREAM_EVAL_INTERVAL = _coerce_positive_int(
        downstream_eval_interval_value,
        field_name="evaluation.downstream.eval_interval",
    )
    EVAL_ITERS = int(os.environ.get("OLMO_EVAL_ITERS", 10))
    EVAL_DATA_WORK_DIR = os.environ.get("EVAL_DATA_WORK_DIR", "olmo2-streaming/tmp/eval_dataset_cache")

    ENABLE_WANDB = bool(os.environ.get("WANDB_API_KEY"))
    ENABLE_COMET = False

    if yaml_path is not None:
        embedding_injection = _build_embedding_injection_config_from_yaml(
            yaml_path,
            warmup_tokens=WARMUP_TOKENS,
            global_batch_size_samples=GLOBAL_BATCH_SIZE_SAMPLES,
            seq_len=SEQ_LEN,
        )
    else:
        embedding_injection = _build_embedding_injection_config_from_env(
            warmup_tokens=WARMUP_TOKENS,
            global_batch_size_samples=GLOBAL_BATCH_SIZE_SAMPLES,
            seq_len=SEQ_LEN,
            streaming_tokenizer_model=resolved.streaming_tokenizer_model,
        )

    init_std_value = model_yaml.get("init_std")
    derived_init_std: Optional[float] = None
    if init_std_value is not None:
        derived_init_std = float(init_std_value)
    elif yaml_path is not None:
        derived_init_std = _derive_default_init_std(HIDDEN_SIZE)


    cfg = build_config(
        run_name=resolved.run_name,
        streaming_data_path=list(resolved.streaming_data_path),
        streaming_tokenizer_model=resolved.streaming_tokenizer_model,
        streaming_ckpt_path=resolved.streaming_ckpt_path,
        save_root=resolved.save_root,
        seq_len=SEQ_LEN,
        micro_batch_size=MICRO_BATCH_SIZE,
        global_batch_size_samples=GLOBAL_BATCH_SIZE_SAMPLES,
        train_tokens=LR_DECAY_TOKENS,
        warmup_fraction=WARMUP_FRACTION,
        warmup_tokens=WARMUP_TOKENS,
        save_tokens=SAVE_TOKENS,
        lr=LR,
        min_lr=MIN_LR,
        weight_decay=WEIGHT_DECAY,
        beta1=BETA1,
        beta2=BETA2,
        num_layers=NUM_LAYERS,
        hidden_size=HIDDEN_SIZE,
        ffn_hidden_size=FFN_HIDDEN_SIZE,
        num_attn_heads=NUM_ATTN_HEADS,
        num_query_groups=NUM_QUERY_GROUPS,
        qk_norm=QK_NORM,
        rope_theta=ROPE_THETA,
        layer_norm_eps=LAYER_NORM_EPS,
        dp_type=DP_TYPE,
        param_dtype=PARAM_DTYPE,
        reduce_dtype=REDUCE_DTYPE,
        grad_clip=GRAD_CLIP,
        ac_mode=AC_MODE,
        enable_wandb=ENABLE_WANDB,
        enable_comet=ENABLE_COMET,
        downstream_enabled=DOWNSTREAM_ENABLED,
        downstream_eval_interval=DOWNSTREAM_EVAL_INTERVAL,
        eval_iters=EVAL_ITERS,
        log_interval=LOG_INTERVAL,
        text_chunk_queue_size=TEXT_CHUNK_QUEUE_SIZE,
        text_chunk_size=TEXT_CHUNK_SIZE,
        prefetch_queue_size=PREFETCH_QUEUE_SIZE,
        use_token_column=USE_TOKEN_COLUMN,
        pack_method=PACK_METHOD,
        init_std=derived_init_std,
        embedding_injection=embedding_injection,
        eval_data_work_dir=EVAL_DATA_WORK_DIR,
    ).merge(overrides)

    launch_meta = resolved.as_metadata()

    if print_resolved_config:
        print(
            json.dumps(
                {
                    "launch_meta": launch_meta,
                    "config": cfg.as_config_dict(),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    _log_resolved_launch_metadata(resolved)

    
    
    # Log model diagnostics before training starts
    log_model_diagnostics(cfg)
    
    # Set RNG states on all devices (align with other scripts)
    seed_all(cfg.init_seed)
    
    model = cfg.model.build(init_device="meta")
    if cfg.model.embedding_injection is not None:
        injection_cfg = cfg.model.embedding_injection
        injection_version = injection_cfg.mode or "None"
        injection_targets, qkvo_targets, h_target_enabled = _resolve_injection_target_state_from_config(
            injection_cfg
        )
        print(f"\n{'='*80}")
        print(f"Embedding injection enabled (version: {injection_version}):")
        print(f"  H layer indices: {injection_cfg.h_layers}")
        print(f"  QK layer indices: {injection_cfg.qk_layers}")
        print(f"  Q layer indices: {injection_cfg.q_layers}")
        print(f"  K layer indices: {injection_cfg.k_layers}")
        print(f"  V layer indices: {injection_cfg.v_layers}")
        print(f"  O layer indices: {injection_cfg.o_layers}")
        if injection_version == "X-gram":
            print(f"  Targets: {injection_targets}")
            if h_target_enabled:
                print(f"  H path: h = h + (1/√n) * Σ (gate_j * depth_scale * warmup) * E_j")
            if qkvo_targets:
                print(f"  Attention targets: {qkvo_targets}")
                print(f"  Attention path: target = target + (1/√n) * Σ (gate_j * depth_scale * warmup) * E_j")
        elif injection_version == "Engram":
            print(f"  Gating: internal gate (EngramOneGramModule)")
            print(f"  Injection formula: h = h + (1/√n) * Σ engram_j(input_ids, h)")
        elif injection_version == "Retoken":
            print(f"  Gating: ReToken modulation vector")
            print(f"  Injection location: before the FFN residual (elementwise scaling)")
            print(f"  Modulation formula: Δm = Δm ⊙ (1 + s^l ⊙ Norm(E^l[x]))")
        else:
            print(f"  (Unknown version; skipping detailed diagnostics)")
        print(f"{'='*80}\n")

    train_module = cfg.train_module.build(model)
    data_loader = cfg.data_loader.build(dp_process_group=train_module.dp_process_group)
    trainer = cfg.trainer.build(train_module, data_loader)

    config_saver = trainer.callbacks.get("config_saver")
    if isinstance(config_saver, ConfigSaverCallback):
        config_saver.config = cfg.as_config_dict()
        config_saver.config["_launch_meta"] = launch_meta
        print("write config...")
        try:
            trainer.write_file("config.json", json.dumps(config_saver.config))
        except Exception as exc:
            log.warning("Failed to write root config.json: %s", exc)

    # Keep TensorBoard logs under the save folder by default.
    for name, callback in list(trainer.callbacks.items()):
        if name == "tensorboard" and isinstance(callback, TensorBoardCallback):
            callback.log_dir = os.path.join(trainer.save_folder, "tensorboard")


    trainer.fit()
    
    log = logging.getLogger(__name__)
    log.info("Starting cleanup process...")
    
    log.info("Closing wandb...")
    if "wandb" in trainer.callbacks:
        wandb_callback = trainer.callbacks["wandb"]
        if hasattr(wandb_callback, "run") and wandb_callback.run is not None:
            try:
                wandb_callback.run.finish()
                log.info("wandb run finished successfully")
            except Exception as e:
                log.warning(f"Failed to finish wandb run: {e}")
    
    log.info("Closing data loader...")
    iterator = getattr(data_loader, "_iterator", None)
    if iterator is not None:
        shutdown_workers = getattr(iterator, "_shutdown_workers", None)
        if callable(shutdown_workers):
            try:
                shutdown_workers()
                log.info("Data loader workers shutdown successfully")
            except Exception as e:
                log.warning(f"Failed to shutdown data loader workers: {e}")
        try:
            delattr(data_loader, "_iterator")
        except AttributeError:
            pass

    inner_loader = getattr(data_loader, "data_loader", None)
    if inner_loader is not None:
        for close_attr in ("close", "shutdown", "stop"):
            close_fn = getattr(inner_loader, close_attr, None)
            if callable(close_fn):
                try:
                    close_fn()
                    log.info(f"TokenStreamDataLoader.{close_attr}() called")
                    break
                except Exception as e:
                    log.warning(f"Failed to call TokenStreamDataLoader.{close_attr}(): {e}")
    
    log.info("Deleting large objects...")
    del trainer
    del train_module
    del data_loader
    
    log.info("Running garbage collection...")
    gc.collect()
    

    if torch.cuda.is_available():
        log.info("Clearing CUDA cache...")
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        

        for device_id in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(device_id)
            torch.cuda.reset_accumulated_memory_stats(device_id)

        ipc_collect = getattr(torch.cuda, "ipc_collect", None)
        if callable(ipc_collect):
            try:
                ipc_collect()
                log.info("torch.cuda.ipc_collect() executed")
            except Exception as e:
                log.warning(f"torch.cuda.ipc_collect() failed: {e}")
    

    gc.collect()
    

    if dist.is_available() and dist.is_initialized():
        log.info("Waiting for all ranks to finish cleanup barrier...")
        try:
            dist.barrier()
        except Exception as e:
            log.warning(f"Cleanup barrier failed: {e}")

    log.info("Waiting for async operations to complete...")
    time.sleep(2)
    
    log.info("Cleanup process completed successfully")
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train OLMo model with streaming data")
    parser.add_argument("--run-name", type=str, default=None, help="Run name for this training")
    parser.add_argument("--save-root", type=str, default=None, help="Root directory under which checkpoints are saved in a run_name subdirectory")
    parser.add_argument("--micro-batch-size", type=int, default=None, help="Micro batch size per GPU")
    parser.add_argument("--streaming-data-path", nargs="*", default=None, help="Streaming data paths")
    parser.add_argument("--streaming-tokenizer-model", type=str, default=None, help="Tokenizer model path")
    parser.add_argument("--streaming-ckpt-path", type=str, default=None, help="Streaming checkpoint path")
    parser.add_argument(
        "--streaming-text-chunk-queue-size",
        type=int,
        default=None,
        help="Size of the streaming text chunk queue (overrides env TEXT_CHUNK_QUEUE_SIZE)",
    )
    parser.add_argument(
        "--streaming-text-chunk-size",
        type=int,
        default=None,
        help="Size of each streaming text chunk (overrides env TEXT_CHUNK_SIZE)",
    )
    parser.add_argument(
        "--streaming-prefetch-queue-size",
        type=int,
        default=None,
        help="Prefetch queue size for streaming data (overrides env PREFETCH_QUEUE_SIZE)",
    )
    parser.add_argument(
        "--streaming-use-token-column",
        type=str,
        default=None,
        help="Token column name to use for streaming data (overrides env STREAMING_USE_TOKEN_COLUMN)",
    )
    parser.add_argument(
        "--streaming-pack-method",
        type=str,
        default=None,
        help="Packing method for streaming data (overrides env STREAMING_PACK_METHOD)",
    )
    parser.add_argument(
        "--streaming-pack-system",
        type=str,
        default=None,
        help="Deprecated alias for specifying the streaming pack implementation (e.g. 'native').",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to the public YAML configuration file.",
    )
    parser.add_argument(
        "--embedding-injection-yaml",
        type=str,
        default=None,
        help="Deprecated alias for --config.",
    )
    parser.add_argument(
        "--launcher-mode",
        choices=["plain", "shell_compatible"],
        default="plain",
        help="Select launcher-style default resolution behavior.",
    )
    parser.add_argument(
        "--print-resolved-config",
        action="store_true",
        help="Print resolved launch metadata and config, then exit without training.",
    )
    
    args, unknown = parser.parse_known_args()
    if args.streaming_pack_system:
        pack_system = args.streaming_pack_system.lower()
        if pack_system == "native":
            args.streaming_pack_method = "native:truncate"
        elif pack_system in {"none", "null"}:
            args.streaming_pack_method = "none"
        else:
            raise OLMoConfigurationError(
                f"Unsupported streaming pack system '{pack_system}'. "
                "Please use --streaming-pack-method to pick an explicit strategy."
            )
    
    overrides = unknown

    prepare_training_environment()
    torch.use_deterministic_algorithms(True)
    try:
        main(
            run_name=args.run_name,
            streaming_data_path=args.streaming_data_path,
            streaming_tokenizer_model=args.streaming_tokenizer_model,
            streaming_ckpt_path=args.streaming_ckpt_path,
            save_root=args.save_root,
            micro_batch_size=args.micro_batch_size,
            streaming_text_chunk_queue_size=args.streaming_text_chunk_queue_size,
            streaming_text_chunk_size=args.streaming_text_chunk_size,
            streaming_prefetch_queue_size=args.streaming_prefetch_queue_size,
            streaming_use_token_column=args.streaming_use_token_column,
            streaming_pack_method=args.streaming_pack_method,
            config_path=args.config,
            embedding_injection_yaml=args.embedding_injection_yaml,
            launcher_mode=args.launcher_mode,
            print_resolved_config=args.print_resolved_config,
            overrides=overrides,
        )
    finally:
        teardown_training_environment()
