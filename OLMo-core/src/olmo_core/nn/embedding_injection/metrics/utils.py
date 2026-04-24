from typing import Union

import torch


def _warmup_scale_to_python_float(warmup_scale: Union[float, torch.Tensor]) -> float:
    if torch.is_tensor(warmup_scale):
        return float(warmup_scale.detach().float().cpu().item())
    return float(warmup_scale)
