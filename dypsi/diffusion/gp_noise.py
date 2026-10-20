# Core component of DyPSI (NeurIPS 2026).
# Adapted from SDIFT (Chen et al., 2025): https://github.com/OceanSTARLab/SDIFT
"""Gaussian-process temporal noise for the joint field+SPF latent diffusion.

The perturbation is correlated along the time mode only: the Cholesky factor
of a temporal Gram matrix (Matern-3/2 by default) is applied to i.i.d.
Gaussian noise along the time mode, leaving channel/spatial modes independent.
"""

from __future__ import annotations

import math

import torch

__all__ = ["matern32_gram", "rbf_gram", "TemporalGPNoise"]

_SQRT3 = math.sqrt(3.0)


def _pairwise_time_diff(t: torch.Tensor) -> torch.Tensor:
    """Pairwise time differences ``t_i - t_j`` from a time grid.

    Args:
        t: Time coordinates, ``(T,)`` for a shared grid or ``(B, T, 1)`` for
            per-sample grids.

    Returns:
        Tensor of shape ``(T, T)`` or ``(B, T, T)``.
    """
    if t.dim() == 1:
        tt = t.view(-1, 1)  # (T, 1)
    elif t.dim() == 3 and t.shape[-1] == 1:
        tt = t  # (B, T, 1)
    else:
        raise ValueError(
            f"Expected time coordinates of shape (T,) or (B, T, 1), got {tuple(t.shape)}"
        )
    return tt - tt.transpose(-1, -2)  # (T, T) or (B, T, T)


def _add_jitter(gram: torch.Tensor, jitter: float) -> torch.Tensor:
    """Add ``jitter * I`` along the last two dimensions for stability."""
    eye = torch.eye(gram.shape[-1], dtype=gram.dtype, device=gram.device)
    return gram + jitter * eye


def matern32_gram(
    t: torch.Tensor,
    length_scale: float = 4.0,
    signal_var: float = 1.0,
    jitter: float = 1e-4,
) -> torch.Tensor:
    """Matern-3/2 Gram matrix over a time grid (DyPSI default).

    ``K(r) = signal_var * (1 + sqrt(3) r / l) exp(-sqrt(3) r / l)`` with
    ``r = |t_i - t_j|`` and ``l = length_scale``.

    Args:
        t: Time coordinates, ``(T,)`` or ``(B, T, 1)``.
        length_scale: Kernel length-scale ``l`` (same units as ``t``).
        signal_var: Signal variance (kernel value at ``r = 0``).
        jitter: Diagonal jitter for numerical stability of the Cholesky
            factorisation.

    Returns:
        Gram matrix of shape ``(T, T)`` or ``(B, T, T)``.
    """
    s = _pairwise_time_diff(t)
    r = s.abs()
    a = _SQRT3 * r / length_scale
    gram = signal_var * (1.0 + a) * torch.exp(-a)
    return _add_jitter(gram, jitter)


def rbf_gram(
    t: torch.Tensor,
    gp_gamma: float = 50.0,
    jitter: float = 1e-5,
) -> torch.Tensor:
    """RBF Gram matrix ``exp(-gamma * (t_i - t_j)^2)`` (SDIFT original kernel, kept as an ablation).

    Args:
        t: Time coordinates, ``(T,)`` or ``(B, T, 1)``.
        gp_gamma: Inverse length-scale coefficient of the RBF kernel.
        jitter: Diagonal jitter for numerical stability.

    Returns:
        Gram matrix of shape ``(T, T)`` or ``(B, T, T)``.
    """
    s = _pairwise_time_diff(t)
    gram = torch.exp(-torch.square(s) * gp_gamma)
    return _add_jitter(gram, jitter)


class TemporalGPNoise:
    """Cholesky-based generator of temporally correlated GP noise.

    The Gram matrix of the chosen temporal kernel and its Cholesky factor are
    precomputed once; :meth:`apply` maps i.i.d. Gaussian samples to
    perturbations correlated along the time mode ``dim=1`` only.

    Attributes:
        gram: Temporal Gram matrix, shape ``(T, T)`` or ``(B, T, T)``.
        chol: Lower-triangular Cholesky factor ``L`` with ``L L^T = gram``.
    """

    def __init__(
        self,
        t_grid: torch.Tensor,
        kernel: str = "matern32",
        length_scale: float = 4.0,
        signal_var: float = 1.0,
        gp_gamma: float = 50.0,
    ) -> None:
        """Build the temporal Gram matrix and its Cholesky factor.

        Args:
            t_grid: Time coordinates, ``(T,)`` for a shared grid or ``(B, T, 1)``
                for per-sample grids.
            kernel: ``"matern32"`` (DyPSI default) or ``"rbf"`` (SDIFT
                original, ablation baseline).
            length_scale: Matern-3/2 length-scale (frames, if ``t_grid`` is
                given in frame indices).
            signal_var: Matern-3/2 signal variance.
            gp_gamma: RBF inverse length-scale (only used for ``kernel="rbf"``).
        """
        if kernel == "matern32":
            self.gram = matern32_gram(t_grid, length_scale=length_scale, signal_var=signal_var)
        elif kernel == "rbf":
            self.gram = rbf_gram(t_grid, gp_gamma=gp_gamma)
        else:
            raise ValueError(f"Unknown temporal GP kernel: {kernel!r}")
        self.chol = torch.linalg.cholesky(self.gram)  # (T, T) or (B, T, T)

    def apply(self, xi: torch.Tensor) -> torch.Tensor:
        """Correlate i.i.d. noise along the time mode.

        Args:
            xi: I.i.d. standard Gaussian noise of shape ``(B, T, ...)``, where
                ``T`` must match the time dimension of the Gram matrix.

        Returns:
            Temporally correlated noise with the same shape as ``xi``.
        """
        B, T = xi.shape[0], xi.shape[1]
        flat = xi.reshape(B, T, -1)  # (B, T, F)
        chol = self.chol.to(device=xi.device, dtype=xi.dtype)
        if chol.dim() == 2:
            out = chol @ flat  # (T, T) @ (B, T, F) -> (B, T, F)
        else:
            out = torch.einsum("bts,bsf->btf", chol, flat)
        return out.reshape(xi.shape)

    def sample(
        self,
        shape: tuple,
        device=None,
        dtype=None,
    ) -> torch.Tensor:
        """Draw temporally correlated standard GP noise.

        Args:
            shape: Target shape ``(B, T, ...)``; ``shape[1]`` must equal the
                time dimension of the Gram matrix.
            device: Target device (defaults to the Cholesky factor's device).
            dtype: Target dtype (defaults to the Cholesky factor's dtype).

        Returns:
            Correlated noise tensor of shape ``shape``.
        """
        if device is None:
            device = self.chol.device
        if dtype is None:
            dtype = self.chol.dtype
        xi = torch.randn(shape, device=device, dtype=dtype)
        return self.apply(xi)
