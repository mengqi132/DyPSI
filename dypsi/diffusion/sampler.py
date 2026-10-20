# Core component of DyPSI (NeurIPS 2026).
# Adapted from SDIFT (Chen et al., 2025, built on EDM by Karras et al., 2022): https://github.com/OceanSTARLab/SDIFT
"""Conditional posterior sampler for the joint field+SPF latent.

Second-order Heun integration of the probability-flow ODE with soft anchoring
of an observed prefix at every reverse step, plus autoregressive
sliding-window roll-out. ODE accumulation is carried out in float64; the
denoiser is evaluated in float32.
"""

from __future__ import annotations

import torch

from .gp_noise import TemporalGPNoise

__all__ = ["edm_sigma_steps", "conditional_heun_sample", "SlidingWindowGenerator"]


def edm_sigma_steps(
    num_steps: int,
    sigma_min: float,
    sigma_max: float,
    rho: float,
    device,
    dtype=torch.float64,
) -> torch.Tensor:
    """Karras rho-schedule of noise levels.

    Args:
        num_steps: Number of reverse steps.
        sigma_min: Smallest noise level.
        sigma_max: Largest noise level.
        rho: Schedule exponent.
        device: Target device.
        dtype: Schedule dtype (float64 by default for numerical accuracy).

    Returns:
        Noise levels of shape ``(num_steps + 1,)``, decreasing from
        ``sigma_max`` to ``sigma_min`` with a terminal ``0`` appended.
    """
    step_indices = torch.arange(num_steps, dtype=dtype, device=device)
    sigmas = (
        sigma_max ** (1.0 / rho)
        + step_indices / (num_steps - 1) * (sigma_min ** (1.0 / rho) - sigma_max ** (1.0 / rho))
    ) ** rho
    return torch.cat([sigmas, torch.zeros_like(sigmas[:1])])  # t_N = 0


@torch.no_grad()
def conditional_heun_sample(
    denoise_fn,
    shape: tuple,
    t_grid: torch.Tensor,
    num_steps: int = 50,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    rho: float = 7.0,
    observed: "torch.Tensor | None" = None,
    gp_noise: "TemporalGPNoise | None" = None,
    history_emb=None,
    anchor: str = "linear",
    zeta: float = 1.0,
    device=None,
) -> torch.Tensor:
    """Second-order Heun sampler with soft prefix anchoring.

    Args:
        denoise_fn: Callable ``(x, sigma, t_grid, history_emb=None) -> x_0``
            evaluating the preconditioned denoiser. ``x`` is float32, ``sigma``
            is a scalar (0-dim tensor) or per-sample ``(B,)`` tensor.
        shape: Shape ``(B, T, C, ...)`` of the joint latent to generate.
        t_grid: Time labels, shape ``(B, T, 1)``.
        num_steps: Number of Heun steps.
        sigma_min: Smallest noise level of the schedule.
        sigma_max: Largest noise level of the schedule.
        rho: Schedule exponent.
        observed: Optional clean (normalised) observed prefix of shape
            ``(B, K_obs, C, ...)`` with ``K_obs <= T``. If given, the first
            ``K_obs`` frames are softly anchored at every reverse step.
        gp_noise: Optional temporal GP noise generator; if given, the initial
            latent is correlated along the time mode, otherwise i.i.d.
            Gaussian.
        history_emb: Optional history embedding forwarded to ``denoise_fn``.
        anchor: Anchoring annealing strategy: ``"linear"`` (default,
            ``alpha_r`` decreases linearly from 1 to 0) or ``"harmonic"``
            (legacy ablation, ``alpha_r = zeta / (r + 1)``).
        zeta: Anchoring strength for ``anchor="harmonic"``.
        device: Sampling device (defaults to ``t_grid.device``).

    Returns:
        Generated joint latent ``x_0`` of shape ``shape``, float32.
    """
    if anchor not in ("linear", "harmonic"):
        raise ValueError(f"Unknown anchor schedule: {anchor!r}")
    if device is None:
        device = t_grid.device
    B, T = shape[0], shape[1]

    k_obs = 0
    if observed is not None:
        observed = observed.to(device=device, dtype=torch.float64)
        if observed.shape[0] != B or observed.shape[2:] != shape[2:]:
            raise ValueError(
                f"observed shape {tuple(observed.shape)} incompatible with {tuple(shape)}"
            )
        k_obs = observed.shape[1]
        if k_obs > T:
            raise ValueError(f"observed prefix length {k_obs} exceeds T={T}")

    # Initial latent at sigma_max: temporally correlated or i.i.d.
    if gp_noise is not None:
        latents = gp_noise.sample(shape, device=device, dtype=torch.float64)
    else:
        latents = torch.randn(shape, device=device, dtype=torch.float64)

    sigma_steps = edm_sigma_steps(num_steps, sigma_min, sigma_max, rho, device=device)
    t_grid32 = t_grid.to(device=device, dtype=torch.float32)

    x_next = latents * sigma_steps[0]  # x_T ~ N(0, sigma_max^2 K_time (x) I)

    for i, (sig_cur, sig_next) in enumerate(zip(sigma_steps[:-1], sigma_steps[1:])):
        # Euler step (denoiser evaluated in float32, accumulated in float64).
        denoised = denoise_fn(
            x_next.to(torch.float32), sig_cur, t_grid32, history_emb=history_emb
        ).to(torch.float64)
        d_cur = (x_next - denoised) / sig_cur
        x_next = x_next + (sig_next - sig_cur) * d_cur

        # Second-order correction (Heun).
        if i < num_steps - 1:
            denoised_next = denoise_fn(
                x_next.to(torch.float32), sig_next, t_grid32, history_emb=history_emb
            ).to(torch.float64)
            d_prime = (x_next - denoised_next) / sig_next
            x_next = x_next + (sig_next - sig_cur) * (0.5 * d_cur + 0.5 * d_prime)

            # Soft prefix anchoring: pull the observed frames toward the
            # observations noised at the current level sigma_i.
            if k_obs > 0:
                if anchor == "linear":
                    alpha_r = 1.0 - i / (num_steps - 1)
                else:  # "harmonic" (legacy ablation)
                    alpha_r = zeta / (i + 1)
                if alpha_r > 0:
                    eps = torch.randn(
                        observed.shape, device=device, dtype=torch.float64
                    )
                    obs_noisy = observed + sig_cur * eps
                    x_next[:, :k_obs] = (
                        (1.0 - alpha_r) * x_next[:, :k_obs] + alpha_r * obs_noisy
                    )

    return x_next.float()


class SlidingWindowGenerator:
    """Autoregressive generation of sequences longer than ``seg_len``.

    The diffusion model generates segments of fixed length ``seg_len``. After
    each segment, its last ``k_obs`` frames are used as the observed prefix
    for conditioning the next segment, so trajectories of arbitrary length
    can be rolled out autoregressively.
    """

    def __init__(
        self,
        denoise_fn,
        seg_len: int,
        k_obs: int,
        sample_kwargs: dict,
    ) -> None:
        """Initialise the generator.

        Args:
            denoise_fn: Callable denoiser, see :func:`conditional_heun_sample`.
            seg_len: Segment length generated per sampling call.
            k_obs: Number of trailing frames of each segment reused as the
                observed prefix of the next segment. Must satisfy
                ``0 < k_obs < seg_len``.
            sample_kwargs: Extra keyword arguments forwarded to
                :func:`conditional_heun_sample`. May contain ``t_grid`` to
                override the default ``linspace(0, 1, seg_len)`` grid;
                ``observed`` is managed internally and is ignored.
        """
        if not 0 < k_obs < seg_len:
            raise ValueError(f"Require 0 < k_obs < seg_len, got k_obs={k_obs}, seg_len={seg_len}")
        self.denoise_fn = denoise_fn
        self.seg_len = seg_len
        self.k_obs = k_obs
        self.sample_kwargs = dict(sample_kwargs)
        self.sample_kwargs.pop("observed", None)  # managed internally
        self._t_grid_override = self.sample_kwargs.pop("t_grid", None)

    def _make_t_grid(self, B: int, device) -> torch.Tensor:
        """Return the per-segment time grid, shape ``(B, seg_len, 1)``."""
        t_grid = self._t_grid_override
        if t_grid is None:
            t_grid = torch.linspace(0, 1, self.seg_len, device=device).view(1, -1, 1)
        if t_grid.dim() == 1:
            t_grid = t_grid.view(1, -1, 1)
        if t_grid.shape[0] != B:
            t_grid = t_grid[:1].repeat(B, 1, 1)
        return t_grid.to(device)

    def generate(self, init_prefix: torch.Tensor, total_len: int) -> torch.Tensor:
        """Roll out a sequence of length ``total_len`` autoregressively.

        The first segment is conditioned on ``init_prefix`` and contributes
        all ``seg_len`` frames; every subsequent segment is conditioned on the
        last ``k_obs`` frames of the previous segment and contributes only its
        ``seg_len - k_obs`` new frames.

        Args:
            init_prefix: Initial observed prefix, shape ``(B, k_obs, C, ...)``.
            total_len: Total number of frames to generate.

        Returns:
            Generated sequence of shape ``(B, total_len, C, ...)``, float32.
        """
        if init_prefix.shape[1] != self.k_obs:
            raise ValueError(
                f"init_prefix must contain k_obs={self.k_obs} frames, "
                f"got {init_prefix.shape[1]}"
            )
        B = init_prefix.shape[0]
        device = init_prefix.device
        seg_shape = (B, self.seg_len, *init_prefix.shape[2:])
        t_grid = self._make_t_grid(B, device)

        segments = []
        prefix = init_prefix
        generated = 0
        call_kwargs = dict(self.sample_kwargs)
        call_kwargs.setdefault("device", device)
        while generated < total_len:
            seg = conditional_heun_sample(
                self.denoise_fn,
                seg_shape,
                t_grid,
                observed=prefix,
                **call_kwargs,
            )
            if not segments:
                segments.append(seg)
                generated += self.seg_len
            else:
                segments.append(seg[:, self.k_obs:])
                generated += self.seg_len - self.k_obs
            prefix = seg[:, -self.k_obs:]

        return torch.cat(segments, dim=1)[:, :total_len]
