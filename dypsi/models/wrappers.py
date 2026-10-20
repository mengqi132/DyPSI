# Denoiser wrappers for latents with more than two spatial rank modes.
# Core component of DyPSI (NeurIPS 2026).
"""Adapts the 2D ``Spatial_temporal_UNet`` to higher-mode latents.

``FoldSpatialModeUNet`` folds the last spatial rank mode (R4) into the
channel dimension by a pure re-indexing: a latent of shape
``(B, T, C, R2, R3, R4)`` is processed by the 2D U-Net as
``(B, T, C * R4, R2, R3)`` (folded channel index ``c * R4 + r4``) and
unfolded back with the inverse permutation afterwards.
"""

from __future__ import annotations

import torch
from torch import nn

__all__ = ["FoldSpatialModeUNet"]


class FoldSpatialModeUNet(nn.Module):
    """Wraps a 2D ``Spatial_temporal_UNet`` to accept 3-spatial-mode latents.

    ``x``: ``(B, T, C, R2, R3, R4)``, folded to ``(B, T, C * R4, R2, R3)``
    for the base U-Net, then unfolded back. ``noise_labels`` /
    ``time_labels`` / ``class_labels`` are forwarded unchanged. The base
    U-Net must be built with ``in_channels == out_channels == C * R4`` and
    ``img_resolution`` matching ``R2`` (= ``R3``).

    Args:
        base_unet: A ``Spatial_temporal_UNet`` (or any module with the same
            forward signature) operating on ``(B, T, C * R4, R2, R3)``.
        n_spatial_modes: Number of spatial rank modes of the incoming latent;
            must be 3 (the wrapper exists exactly for this case).
    """

    def __init__(self, base_unet: nn.Module, n_spatial_modes: int = 3) -> None:
        super().__init__()
        if n_spatial_modes != 3:
            raise ValueError(
                f"FoldSpatialModeUNet expects n_spatial_modes=3, got {n_spatial_modes}"
            )
        self.base_unet = base_unet
        self.n_spatial_modes = n_spatial_modes

    @staticmethod
    def fold(x: torch.Tensor) -> torch.Tensor:
        """(B, T, C, R2, R3, R4) -> (B, T, C * R4, R2, R3)."""
        B, T, C, R2, R3, R4 = x.shape
        x = x.permute(0, 1, 2, 5, 3, 4)          # (B, T, C, R4, R2, R3)
        return x.reshape(B, T, C * R4, R2, R3)

    @staticmethod
    def unfold(y: torch.Tensor, C: int, R4: int) -> torch.Tensor:
        """(B, T, C * R4, R2, R3) -> (B, T, C, R2, R3, R4)."""
        B, T, _, R2, R3 = y.shape
        y = y.view(B, T, C, R4, R2, R3)
        return y.permute(0, 1, 2, 4, 5, 3)       # (B, T, C, R2, R3, R4)

    def forward(self, x: torch.Tensor, noise_labels: torch.Tensor,
                time_labels: torch.Tensor,
                class_labels: "torch.Tensor | None" = None) -> torch.Tensor:
        if x.dim() != 6:
            raise ValueError(
                f"FoldSpatialModeUNet expects x of shape (B, T, C, R2, R3, R4), "
                f"got {tuple(x.shape)}"
            )
        C, R4 = x.shape[2], x.shape[5]
        out = self.base_unet(self.fold(x), noise_labels, time_labels, class_labels)
        return self.unfold(out, C, R4)
