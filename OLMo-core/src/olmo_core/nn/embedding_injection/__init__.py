"""
Embedding injection runtime helpers.

This package contains the mode-specific builders and per-block runtime helpers
used by the current open-source X-gram / Engram / Retoken / Mort implementation.
"""

from .engram import (
    EngramModule,
    apply_engram_pre_block,
    build_engram_modules,
)
from .metrics import _warmup_scale_to_python_float, static_injection_logger, static_o_injection_logger
from .mort import (
    apply_mort_sparse_injection,
    build_mort_modules,
    init_mort_modules,
    prepare_mort_block_kwargs,
)
from .retoken import (
    build_retoken_modules,
    init_retoken_modules,
    prepare_retoken_block_kwargs,
)
from .runtime import InjectionBlockContext, InjectionBlockResult
from .xgram import build_xgram_modules, prepare_xgram_block_kwargs

__all__ = [
    "EngramModule",
    "InjectionBlockContext",
    "InjectionBlockResult",
    "_warmup_scale_to_python_float",
    "apply_engram_pre_block",
    "apply_mort_sparse_injection",
    "build_engram_modules",
    "build_mort_modules",
    "build_retoken_modules",
    "build_xgram_modules",
    "init_mort_modules",
    "init_retoken_modules",
    "prepare_mort_block_kwargs",
    "prepare_retoken_block_kwargs",
    "prepare_xgram_block_kwargs",
    "static_injection_logger",
    "static_o_injection_logger",
]
