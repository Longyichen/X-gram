from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLUShortConv(nn.Module):
    """
    Depthwise shortconv with SwiGLU-style gating: conv_content * silu(conv_gate).
    Expects input shape (B, D, T).
    """

    def __init__(
        self,
        d_model: int,
        kernel_size: int,
        *,
        device: Optional[torch.device],
        dtype: Optional[torch.dtype],
    ):
        super().__init__()
        self.conv_content = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=kernel_size,
            padding=0,
            groups=d_model,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.conv_gate = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=kernel_size,
            padding=0,
            groups=d_model,
            bias=True,
            device=device,
            dtype=dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_content(x) * F.silu(self.conv_gate(x))


