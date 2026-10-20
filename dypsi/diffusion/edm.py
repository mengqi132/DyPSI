# Adapted from SDIFT (Chen et al., 2025, built on EDM by Karras et al., 2022): https://github.com/OceanSTARLab/SDIFT
# Core component of DyPSI (NeurIPS 2026).
"""EDM trainer for the joint field+SPF latent of DyPSI.

Implements the EDM noise sampling, loss weighting, preconditioning, and a
ramped EMA of the network weights; optionally perturbs with temporally
correlated GP noise instead of i.i.d. Gaussian noise.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass

import torch

from .gp_noise import TemporalGPNoise

__all__ = ["EDMConfig", "JointEDM"]


@dataclass
class EDMConfig:
    """Configuration of the joint EDM trainer.

    Attributes:
        sigma_min: Smallest supported noise level.
        sigma_max: Largest supported noise level.
        rho: Exponent of the Karras sigma schedule (used at sampling time).
        sigma_data: Expected data standard deviation used by the EDM
            preconditioning and loss weighting.
        P_mean: Mean of the log-normal sigma sampling distribution.
        P_std: Std of the log-normal sigma sampling distribution.
        ema_halflife_kimg: EMA half-life in thousands of training samples.
        ema_rampup_ratio: Warm-up ramp for the EMA half-life; ``None``
            disables the ramp-up.
        gt_guide_type: Regression loss against the clean latent, ``"l2"`` or
            ``"l1"``.
    """

    sigma_min: float = 0.002
    sigma_max: float = 80.0
    rho: float = 7.0
    sigma_data: float = 0.5
    P_mean: float = -1.2
    P_std: float = 1.2
    ema_halflife_kimg: float = 500.0
    ema_rampup_ratio: float = 0.05
    gt_guide_type: str = "l2"  # or "l1"


class JointEDM:
    """EDM trainer for the joint field+SPF latent.

    The wrapped model is a ``Spatial_temporal_UNet`` with signature
    ``forward(x, noise_labels, time_labels, class_labels=None)`` operating on
    joint latents of shape ``(B, T, C, R2, R3)``.

    Attributes:
        model: The trainable denoiser network.
        ema: EMA copy of ``model``, used for sampling.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        cfg: EDMConfig,
        batch_size: int,
        gp_noise: "TemporalGPNoise | None" = None,
        device: torch.device | str = "cpu",
    ) -> None:
        """Initialise the trainer.

        Args:
            model: Denoiser network ``forward(x, noise_labels, time_labels,
                class_labels=None)``.
            cfg: EDM hyper-parameters.
            batch_size: Training batch size (used by the EMA update rule).
            gp_noise: Optional temporal GP noise generator; if ``None``,
                i.i.d. Gaussian noise is used.
            device: Device the model and EMA copy are moved to.
        """
        self.cfg = cfg
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.gp_noise = gp_noise

        self.model = model.to(self.device)
        self.ema = copy.deepcopy(self.model)
        self.ema.eval()
        self.ema.requires_grad_(False)
        self.model.train()

    @staticmethod
    def _broadcast(c: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Reshape a per-sample coefficient ``(B,)`` to broadcast over ``x``."""
        return c.view(-1, *([1] * (x.ndim - 1)))

    def model_forward_wrapper(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        t: torch.Tensor,
        history_emb: "torch.Tensor | None" = None,
        use_ema: bool = False,
    ) -> torch.Tensor:
        """Apply the EDM preconditioning and evaluate the denoiser.

        Computes ``D_theta(x; sigma, h) = c_skip * x + c_out * F_theta(c_in * x;
        c_noise, t, h)`` where ``h`` is the optional history embedding.

        Args:
            x: Noisy joint latent, shape ``(B, T, C, R2, R3)``.
            sigma: Noise level per sample, shape ``(B,)``.
            t: Time labels, shape ``(B, T, 1)``.
            history_emb: Optional history embedding of shape ``(B, D)``,
                forwarded to the UNet as ``class_labels``.
            use_ema: If ``True``, evaluate the EMA weights.

        Returns:
            Denoised joint latent with the same shape as ``x``.
        """
        sigma = torch.as_tensor(sigma, device=x.device, dtype=x.dtype)
        if sigma.dim() == 0:
            sigma = sigma.expand(x.shape[0])
        sigma = sigma.clamp_min(self.cfg.sigma_min)
        t = t.to(device=x.device, dtype=x.dtype)

        sigma_data = self.cfg.sigma_data
        c_skip = sigma_data**2 / (sigma**2 + sigma_data**2)
        c_out = sigma * sigma_data / (sigma**2 + sigma_data**2).sqrt()
        c_in = 1.0 / (sigma_data**2 + sigma**2).sqrt()
        c_noise = sigma.log() / 4.0

        x_in = self._broadcast(c_in, x) * x
        noise_labels = c_noise.view(-1, 1, 1).repeat(1, t.shape[1], 1)  # (B, T, 1)

        net = self.ema if use_ema else self.model
        model_output = net(x_in, noise_labels, t, history_emb)

        return self._broadcast(c_skip, x) * x + self._broadcast(c_out, x) * model_output

    def train_step(
        self,
        signals: torch.Tensor,
        t: torch.Tensor,
        history_emb: "torch.Tensor | None" = None,
    ) -> torch.Tensor:
        """One EDM training step on a batch of clean joint latents.

        Args:
            signals: Clean joint latent, shape ``(B, T, C, R2, R3)``.
            t: Time labels, shape ``(B, T, 1)``.
            history_emb: Optional history embedding of shape ``(B, D)``.

        Returns:
            Scalar weighted denoising loss.
        """
        signals = signals.to(self.device)
        t = t.to(self.device)
        B = signals.shape[0]

        # Log-normal noise-level sampling.
        rnd_normal = torch.randn(B, device=signals.device)
        sigma = (rnd_normal * self.cfg.P_std + self.cfg.P_mean).exp()  # (B,)
        weight = (sigma**2 + self.cfg.sigma_data**2) / (sigma * self.cfg.sigma_data) ** 2

        # Temporally correlated perturbation or i.i.d. ablation.
        if self.gp_noise is not None:
            noise = self.gp_noise.sample(signals.shape, device=signals.device, dtype=signals.dtype)
        else:
            noise = torch.randn_like(signals)

        n = self._broadcast(sigma, signals) * noise
        denoised_pred = self.model_forward_wrapper(
            signals + n, sigma, t, history_emb=history_emb, use_ema=False
        )

        if self.cfg.gt_guide_type == "l2":
            per_entry = (denoised_pred - signals) ** 2
        elif self.cfg.gt_guide_type == "l1":
            per_entry = torch.abs(denoised_pred - signals)
        else:
            raise NotImplementedError(
                f"gt_guide_type {self.cfg.gt_guide_type!r} not implemented"
            )
        loss = self._broadcast(weight, signals) * per_entry
        return loss.mean()

    def update_ema(self, step: int) -> None:
        """Update the EMA weights after optimisation step ``step``.

        Args:
            step: Current optimisation step (0-based), used for the ramp-up.
        """
        ema_halflife_nimg = self.cfg.ema_halflife_kimg * 1000.0
        if self.cfg.ema_rampup_ratio is not None:
            ema_halflife_nimg = min(
                ema_halflife_nimg,
                step * self.batch_size * self.cfg.ema_rampup_ratio,
            )
        ema_beta = 0.5 ** (self.batch_size / max(ema_halflife_nimg, 1e-8))

        with torch.no_grad():
            for p_ema, p_net in zip(self.ema.parameters(), self.model.parameters()):
                p_ema.copy_(p_net.detach().lerp(p_ema, ema_beta))

    def __call__(
        self,
        x: torch.Tensor,
        sigma: "torch.Tensor | float",
        t: torch.Tensor,
        history_emb: "torch.Tensor | None" = None,
        use_ema: bool = True,
    ) -> torch.Tensor:
        """Denoise ``x`` at noise level ``sigma`` (sampling interface).

        Args:
            x: Noisy joint latent, shape ``(B, T, C, R2, R3)``.
            sigma: Scalar or per-sample ``(B,)`` noise level.
            t: Time labels, shape ``(B, T, 1)``.
            history_emb: Optional history embedding of shape ``(B, D)``.
            use_ema: If ``True`` (default), use the EMA weights.

        Returns:
            Denoised joint latent with the same shape as ``x``.
        """
        if not torch.is_tensor(sigma):
            sigma = torch.as_tensor(sigma)
        if sigma.dim() == 0:
            sigma = sigma * torch.ones(x.shape[0], device=x.device)
        return self.model_forward_wrapper(
            x, sigma, t, history_emb=history_emb, use_ema=use_ema
        )

    def round_sigma(self, sigma):
        """Round a sigma to the nearest supported value (identity hook)."""
        return torch.as_tensor(sigma)

    def state_dict(self) -> dict:
        """Serialise the trainer: model weights, EMA weights and config."""
        return {
            "model": self.model.state_dict(),
            "ema": self.ema.state_dict(),
            "cfg": asdict(self.cfg),
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore model weights, EMA weights and config from ``state``."""
        self.model.load_state_dict(state["model"])
        self.ema.load_state_dict(state["ema"])
        if "cfg" in state and state["cfg"] is not None:
            self.cfg = EDMConfig(**state["cfg"])
