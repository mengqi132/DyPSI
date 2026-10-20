# Utility functions for DyPSI (NeurIPS 2026).
# Partially adapted from SDIFT (Chen et al., 2025): https://github.com/OceanSTARLab/SDIFT
"""Common utilities: reproducibility, metrics, sampling on grids, config IO."""

import random

import numpy as np
import torch
import torch.nn.functional as F

__all__ = [
    "set_seed",
    "relative_l2",
    "bilinear_sample",
    "load_config",
    "get_device",
]


def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy and PyTorch (CPU + CUDA) RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def relative_l2(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Relative L2 error ``||pred - target||_2 / ||target||_2`` (scalar)."""
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred {tuple(pred.shape)} vs target {tuple(target.shape)}")
    return torch.norm(pred - target) / torch.norm(target)


def bilinear_sample(field: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    """Bilinearly sample a spatial field at continuous coordinates.

    Args:
        field: Tensor of shape (B, H, W, C) or (H, W, C).
        coords: Tensor of shape (B, k, 2) or (k, 2) in (x, y) order, in grid
            (pixel) units, not normalized.

    Returns:
        Sampled values of shape (B, k, C).
    """
    squeeze_batch = field.dim() == 3
    if squeeze_batch:
        field = field.unsqueeze(0)  # (1, H, W, C)
    if coords.dim() == 2:
        coords = coords.unsqueeze(0)  # (1, k, 2)
    if field.dim() != 4 or coords.dim() != 3 or coords.shape[-1] != 2:
        raise ValueError(
            f"Expected field (B,H,W,C) and coords (B,k,2); got {tuple(field.shape)} and {tuple(coords.shape)}"
        )
    B, H, W, C = field.shape
    if coords.shape[0] != B:
        raise ValueError(f"Batch mismatch: field B={B}, coords B={coords.shape[0]}")

    coords = coords.to(device=field.device, dtype=field.dtype)
    # Grid units -> [-1, 1] normalized coordinates for grid_sample.
    x_norm = 2.0 * coords[..., 0] / max(W - 1, 1) - 1.0
    y_norm = 2.0 * coords[..., 1] / max(H - 1, 1) - 1.0
    grid = torch.stack([x_norm, y_norm], dim=-1).unsqueeze(2)  # (B, k, 1, 2)

    field_nchw = field.permute(0, 3, 1, 2)  # (B, C, H, W)
    sampled = F.grid_sample(
        field_nchw, grid, mode="bilinear", padding_mode="border", align_corners=True
    )  # (B, C, k, 1)
    return sampled.squeeze(-1).permute(0, 2, 1)  # (B, k, C)


def load_config(path: str) -> dict:
    """Load a YAML configuration file (``.yml`` / ``.yaml``) into a plain dict."""
    import yaml

    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        cfg = {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Top-level YAML node in {path} must be a mapping, got {type(cfg)}")
    return cfg


def get_device(pref: str = "auto") -> torch.device:
    """Resolve a device preference string to a ``torch.device``.

    Args:
        pref: ``"auto"`` (CUDA if available, else CPU), ``"cpu"``,
            ``"cuda"``, ``"cuda:1"``, etc.
    """
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if pref.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Device '{pref}' requested but CUDA is not available.")
    return torch.device(pref)
