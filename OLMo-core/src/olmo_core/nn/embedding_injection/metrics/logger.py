import logging
from typing import Optional, Union

import torch
import torch.distributed as dist
import torch.nn.functional as F

log = logging.getLogger(__name__)
_warned_wandb_missing = False
_warned_wandb_inactive = False


def _warmup_scale_to_python_float(warmup_scale: Union[float, torch.Tensor]) -> float:
    if torch.is_tensor(warmup_scale):
        return float(warmup_scale.detach().float().cpu().item())
    return float(warmup_scale)


def _get_active_wandb():
    global _warned_wandb_inactive, _warned_wandb_missing

    try:
        import wandb  # type: ignore
    except Exception:
        if not _warned_wandb_missing:
            log.warning("wandb not available; skipping injection metrics logging")
            _warned_wandb_missing = True
        return None

    if getattr(wandb, "run", None) is None:
        if not _warned_wandb_inactive:
            log.warning("wandb run is not initialized; skipping injection metrics logging")
            _warned_wandb_inactive = True
        return None

    return wandb


@torch.no_grad()
def static_injection_logger(
    h_prev: torch.Tensor,
    injection_delta: torch.Tensor,
    gate: torch.Tensor,
    lambda_raw: Optional[torch.Tensor],
    layer_idx: int,
    step: Optional[int],
    warmup_scale: float = 1.0,
    log_interval: int = 100,
    eps: float = 1e-6,
) -> None:
    if step is None:
        return
    if log_interval <= 0 or step % log_interval != 0:
        return
    if dist.is_available() and dist.is_initialized():
        try:
            if dist.get_rank() != 0:
                return
        except Exception:
            return
    wandb = _get_active_wandb()
    if wandb is None:
        return

    h_flat = h_prev.detach().reshape(-1, h_prev.shape[-1]).float()
    inj_flat = injection_delta.detach().reshape(-1, injection_delta.shape[-1]).float()
    if h_flat.shape[-1] != inj_flat.shape[-1]:
        return

    h_norm = torch.norm(h_flat, p=2, dim=-1).mean()
    inj_norm = torch.norm(inj_flat, p=2, dim=-1).mean()
    gate_val = gate.detach().float().mean()
    raw_irr = inj_norm / (h_norm + eps)
    effective_irr = torch.abs(gate_val) * inj_norm / (h_norm + eps)
    cos_sim = F.cosine_similarity(h_flat, inj_flat, dim=-1).mean()

    payload = {
        f"analysis_injection/layer_{layer_idx}/h_norm": h_norm.item(),
        f"analysis_injection/layer_{layer_idx}/inj_norm": inj_norm.item(),
        f"analysis_injection/layer_{layer_idx}/irr_raw": raw_irr.item(),
        f"analysis_injection/layer_{layer_idx}/irr_effective": effective_irr.item(),
        f"analysis_injection/layer_{layer_idx}/cos_sim": cos_sim.item(),
        f"analysis_injection/layer_{layer_idx}/gamma": gate_val.item(),
    }
    if lambda_raw is not None:
        lambda_val = lambda_raw.detach().float().mean()
        payload[f"analysis_injection/layer_{layer_idx}/lambda_raw"] = lambda_val.item()
        payload[f"analysis_injection/layer_{layer_idx}/lambda_scaled"] = (
            lambda_val * _warmup_scale_to_python_float(warmup_scale)
        ).item()

    wandb.log(payload, step=step)


if hasattr(torch, "compiler") and hasattr(torch.compiler, "disable"):
    static_injection_logger = torch.compiler.disable(static_injection_logger)  # type: ignore


@torch.no_grad()
def static_o_injection_logger(
    *,
    h_prev: torch.Tensor,
    injection_delta: torch.Tensor,
    gate: torch.Tensor,
    lambda_raw: Optional[torch.Tensor],
    input_embedding: Optional[torch.Tensor],
    layer_idx: int,
    step: Optional[int],
    eps: float = 1e-6,
    warmup_scale: float = 1.0,
    log_interval: int = 100,
) -> None:
    if step is None:
        return
    if log_interval <= 0 or step % log_interval != 0:
        return
    if dist.is_available() and dist.is_initialized():
        try:
            if dist.get_rank() != 0:
                return
        except Exception:
            return
    wandb = _get_active_wandb()
    if wandb is None:
        return

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
    effective_irr = torch.abs(gamma_scalar) * inj_norm / (h_norm + eps)
    cos_sim = F.cosine_similarity(h_flat, inj_flat, dim=-1).mean()

    cos_inj_input = None
    cos_h_input = None
    cos_hplusinj_input = None
    if input_embedding is not None and input_embedding.shape[-1] == h_flat.shape[-1]:
        inp_flat = input_embedding.detach().reshape(-1, input_embedding.shape[-1]).float()
        cos_inj_input = F.cosine_similarity(inj_flat, inp_flat, dim=-1).mean()
        cos_h_input = F.cosine_similarity(h_flat, inp_flat, dim=-1).mean()
        cos_hplusinj_input = F.cosine_similarity(h_with_inj_flat, inp_flat, dim=-1).mean()

    lambda_raw_scalar: Optional[torch.Tensor] = None
    if lambda_raw is not None:
        lambda_flat = lambda_raw.detach().reshape(-1).float()
        lambda_raw_scalar = lambda_flat[0] if lambda_flat.numel() > 0 else torch.tensor(0.0)

    payload = {
        f"analysis_o/layer_{layer_idx}/h_norm": h_norm.item(),
        f"analysis_o/layer_{layer_idx}/inj_norm": inj_norm.item(),
        f"analysis_o/layer_{layer_idx}/irr_raw": raw_irr.item(),
        f"analysis_o/layer_{layer_idx}/irr_effective": effective_irr.item(),
        f"analysis_o/layer_{layer_idx}/cos_sim": cos_sim.item(),
        f"analysis_o/layer_{layer_idx}/gamma": gamma_scalar.item(),
    }
    if cos_inj_input is not None:
        payload[f"analysis_o/layer_{layer_idx}/cos_inj_input"] = cos_inj_input.item()
    if cos_h_input is not None:
        payload[f"analysis_o/layer_{layer_idx}/cos_h_input"] = cos_h_input.item()
    if cos_hplusinj_input is not None:
        payload[f"analysis_o/layer_{layer_idx}/cos_hplusinj_input"] = cos_hplusinj_input.item()
    if lambda_raw_scalar is not None:
        payload[f"analysis_o/layer_{layer_idx}/lambda_raw"] = lambda_raw_scalar.item()
        payload[f"analysis_o/layer_{layer_idx}/lambda_after_warmup"] = (
            lambda_raw_scalar.item() * _warmup_scale_to_python_float(warmup_scale)
        )

    wandb.log(payload, step=step)
