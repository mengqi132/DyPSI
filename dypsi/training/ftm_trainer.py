# Adapted from SDIFT (Chen et al., 2025): https://github.com/OceanSTARLab/SDIFT
# Core component of DyPSI (NeurIPS 2026).
"""Unified Functional Tucker Model (FTM) trainer.

Handles both physical-field data (N, T, H, W) and sensor-position-field (SPF)
data with a shared SIREN basis network and one core tensor per sample; a
single-sample tensor (T, H, W) is treated as N=1. When ``cfg.ranks`` has
length 4 (3D-spatial data), the basis is ``Tensor_inr_4D`` and the cores
have shape (N, T, R1, R2, R3, R4).
"""

import os
from dataclasses import dataclass

import numpy as np
import torch
from torch import optim

from dypsi.models.ftm import Tensor_inr_3D, Tensor_inr_4D, tucker_reconstruct_3d, tucker_reconstruct_4d
from dypsi.utils import get_device

__all__ = ["FTMTrainConfig", "FTMTrainer"]


@dataclass
class FTMTrainConfig:
    """Configuration for FTM pre-training.

    Attributes:
        ranks: Core ranks (channel/modal rank first, then one entry per
            spatial mode): length 3 ``(R1, R2, R3)`` for 2D-spatial data,
            length 4 ``(R1, R2, R3, R4)`` for 3D-spatial data.
        omega: SIREN frequency of the basis networks (higher for spiky SPFs).
        lr: AdamW learning rate.
        weight_decay: AdamW weight decay.
        tv_weight: Weight of the total-variation regularizer on the cores.
        max_iters: Number of optimization iterations.
        eval_interval: Evaluate / checkpoint every this many iterations.
        grad_clip: Max gradient norm (0 disables clipping).
        device: Device preference, see ``dypsi.utils.get_device``.
    """

    ranks: tuple = (2, 48, 48)
    omega: float = 20.0
    lr: float = 2e-4
    weight_decay: float = 1e-6
    tv_weight: float = 1e-7
    max_iters: int = 5000
    eval_interval: int = 200
    grad_clip: float = 1.0
    device: str = "auto"


class FTMTrainer:
    """Trains a shared ``Tensor_inr_3D``/``Tensor_inr_4D`` basis + per-sample cores.

    Args:
        data: Field data of shape (N, T, *spatial) or (T, *spatial), where
            ``len(spatial) == len(cfg.ranks) - 1`` (2 or 3 spatial axes);
            NumPy array or torch tensor.
        cfg: Training configuration.

    Attributes:
        basis_net: The shared SIREN basis network (``Tensor_inr_3D`` for
            3-entry ranks, ``Tensor_inr_4D`` for 4-entry ranks).
        core: Per-sample core tensor of shape (N, T, *ranks), trainable.
    """

    def __init__(self, data: "torch.Tensor | np.ndarray", cfg: FTMTrainConfig):
        self.cfg = cfg
        self.device = get_device(cfg.device)
        self.ranks = tuple(cfg.ranks)
        if len(self.ranks) not in (3, 4):
            raise ValueError(
                f"ranks must have length 3 (2D spatial) or 4 (3D spatial), "
                f"got {self.ranks}"
            )
        n_spatial = len(self.ranks) - 1

        data_t = torch.as_tensor(np.asarray(data) if isinstance(data, np.ndarray) else data,
                                 dtype=torch.float32)
        if data_t.dim() == n_spatial + 1:
            data_t = data_t.unsqueeze(0)  # (T, *spatial) -> (1, T, *spatial)
        if data_t.dim() != n_spatial + 2:
            raise ValueError(
                f"Expected data of shape (N,T,*spatial) or (T,*spatial) with "
                f"{n_spatial} spatial axes, got {tuple(data_t.shape)}"
            )
        self.data = data_t.to(self.device)  # (N, T, *spatial)
        self.N, self.T = self.data.shape[0], self.data.shape[1]
        self.spatial_shape = tuple(self.data.shape[2:])
        self.n_spatial_modes = n_spatial

        # Shared SIREN basis network over (t, y, x[, z]).
        if n_spatial == 2:
            self.basis_net = Tensor_inr_3D(self.ranks, omega=cfg.omega).to(self.device)
        else:
            self.basis_net = Tensor_inr_4D(self.ranks, omega=cfg.omega).to(self.device)

        # Per-sample core tensors, constant-initialized.
        self.core = (torch.ones(self.N, self.T, *self.ranks, device=self.device) * 0.5)
        self.core.requires_grad = True

        params = list(self.basis_net.parameters()) + [self.core]
        self.optimizer = optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

        # Normalized coordinate indices in [0, 1] per mode (time + spatial).
        self.mode_idx = [
            torch.arange(n, device=self.device).float() / max(n - 1, 1)
            for n in (self.T, *self.spatial_shape)
        ]
        # Backwards-compatible aliases for the 2D-spatial case.
        self.t_idx = self.mode_idx[0]
        self.y_idx = self.mode_idx[1]
        self.x_idx = self.mode_idx[2]
        if n_spatial == 3:
            self.z_idx = self.mode_idx[3]

        # Keep H/W attributes for backwards compatibility.
        self.H = self.spatial_shape[0]
        self.W = self.spatial_shape[1]

        self.train_losses: list = []
        self.eval_history: list = []
        self.best_rmse = float("inf")

    # ------------------------------------------------------------------ #
    # Reconstruction
    # ------------------------------------------------------------------ #
    def reconstruct(self, idx: "int | None" = None) -> torch.Tensor:
        """Reconstruct the dense field(s) from cores + basis.

        Args:
            idx: Sample index. If None, reconstruct all samples.

        Returns:
            Tensor of shape (N, T, *spatial) when ``idx`` is None, else
            (T, *spatial).
        """
        self.basis_net.mode = "training"
        core = self.core if idx is None else self.core[idx:idx + 1]
        if self.n_spatial_modes == 2:
            U = self.basis_net.U_net(self.t_idx.unsqueeze(1))  # (T, R1)
            V = self.basis_net.V_net(self.y_idx.unsqueeze(1))  # (H, R2)
            W = self.basis_net.W_net(self.x_idx.unsqueeze(1))  # (W, R3)
            out = tucker_reconstruct_3d(core, U, V, W)  # (B, T, H, W)
        else:
            T_ = self.basis_net.T_net(self.t_idx.unsqueeze(1))  # (T, R1)
            U = self.basis_net.U_net(self.y_idx.unsqueeze(1))   # (H, R2)
            V = self.basis_net.V_net(self.x_idx.unsqueeze(1))   # (W, R3)
            W = self.basis_net.W_net(self.z_idx.unsqueeze(1))   # (Z, R4)
            out = tucker_reconstruct_4d(core, T_, U, V, W)  # (B, T, H, W, Z)
        return out if idx is None else out[0]

    def _tv_loss(self) -> torch.Tensor:
        """Total-variation regularizer on the spatial core modes (R2..)."""
        tv = self.core.new_zeros(())
        for ax in range(3, self.core.dim()):  # skip (N, T, R1) axes
            tv = tv + torch.diff(self.core, dim=ax).abs().sum()
        return self.cfg.tv_weight * tv

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    def train_step(self) -> "tuple[float, float]":
        """Single optimization step over the full dataset.

        Returns:
            (rmse, tv): data RMSE and TV regularizer values (floats).
        """
        self.basis_net.train()
        self.optimizer.zero_grad()

        pred = self.reconstruct()  # (N, T, H, W)
        rmse = torch.sqrt(torch.mean((pred - self.data) ** 2))
        tv = self._tv_loss()
        loss = rmse + tv
        loss.backward()

        if self.cfg.grad_clip and self.cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.basis_net.parameters(), self.cfg.grad_clip)
            torch.nn.utils.clip_grad_norm_(self.core, self.cfg.grad_clip)

        self.optimizer.step()
        return rmse.item(), tv.item()

    def evaluate(self, idx: "int | None" = None) -> dict:
        """Evaluate reconstruction quality without gradient tracking.

        Returns:
            Dict with scalar metrics ``rmse``, ``mae`` and ``relative_l2``.
        """
        self.basis_net.eval()
        with torch.no_grad():
            pred = self.reconstruct(idx)
            gt = self.data if idx is None else self.data[idx]
            rmse = torch.sqrt(torch.mean((pred - gt) ** 2)).item()
            mae = torch.mean(torch.abs(pred - gt)).item()
            rel = (torch.norm(pred - gt) / torch.norm(gt)).item()
        return {"rmse": rmse, "mae": mae, "relative_l2": rel}

    def train(self, save_dir: "str | None" = None, ckpt_name: str = "ftm_best.pt") -> dict:
        """Full training loop with periodic evaluation and best-model saving.

        Args:
            save_dir: Directory for the best checkpoint; None disables saving.
            ckpt_name: Checkpoint filename inside ``save_dir``.

        Returns:
            History dict with keys ``train_rmse``, ``eval`` (list of
            ``(iteration, metrics)``) and ``best_rmse``.
        """
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)

        try:
            from tqdm import tqdm
            iterator = tqdm(range(self.cfg.max_iters), desc="Training FTM")
        except ImportError:
            iterator = range(self.cfg.max_iters)

        n_params = sum(p.numel() for p in self.basis_net.parameters())
        print(f"[FTMTrainer] data {tuple(self.data.shape)}, core {tuple(self.core.shape)}, "
              f"basis params {n_params:,}")

        for it in iterator:
            rmse, tv = self.train_step()
            self.train_losses.append(rmse)
            if hasattr(iterator, "set_postfix"):
                iterator.set_postfix({"RMSE": f"{rmse:.6f}", "TV": f"{tv:.2e}"})

            if (it + 1) % self.cfg.eval_interval == 0 or it == self.cfg.max_iters - 1:
                metrics = self.evaluate()
                self.eval_history.append((it + 1, metrics))
                print(f"[Iter {it + 1}/{self.cfg.max_iters}] RMSE: {metrics['rmse']:.6f}, "
                      f"MAE: {metrics['mae']:.6f}, RelL2: {metrics['relative_l2']:.6f}")
                if metrics["rmse"] < self.best_rmse:
                    self.best_rmse = metrics["rmse"]
                    if save_dir is not None:
                        self.save_checkpoint(os.path.join(save_dir, ckpt_name),
                                             extra={"rmse": self.best_rmse})

        print(f"[FTMTrainer] Done. Best RMSE: {self.best_rmse:.6f}")
        return {
            "train_rmse": list(self.train_losses),
            "eval": list(self.eval_history),
            "best_rmse": self.best_rmse,
        }

    # ------------------------------------------------------------------ #
    # Checkpoints
    # ------------------------------------------------------------------ #
    def save_checkpoint(self, path: str, extra: "dict | None" = None) -> None:
        """Save basis weights + cores in the format consumed at inference.

        The checkpoint contains: ``basis_net`` (state dict), ``core`` (CPU
        tensor (N, T, *ranks)), ``ranks``, ``omega`` and ``data_shape``
        (N, T, *spatial), plus any ``extra`` entries.
        """
        ckpt = {
            "basis_net": self.basis_net.state_dict(),
            "core": self.core.data.cpu(),
            "ranks": tuple(self.ranks),
            "omega": self.cfg.omega,
            "data_shape": (self.N, self.T, *self.spatial_shape),
        }
        if extra:
            ckpt.update(extra)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(ckpt, path)

    @classmethod
    def from_checkpoint(cls, path: str, device: str = "auto") -> "FTMTrainer":
        """Load a trained FTM (basis + cores) for inference / decoding.

        Builds an instance without training data; ``reconstruct`` and the
        ``basis_net`` (sampling mode) are available for decoding.

        Args:
            path: Checkpoint written by :meth:`save_checkpoint`.
            device: Device preference, see ``dypsi.utils.get_device``.
        """
        dev = get_device(device)
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        ranks = tuple(ckpt["ranks"])
        omega = float(ckpt.get("omega", 10.0))

        obj = cls.__new__(cls)
        obj.cfg = FTMTrainConfig(ranks=ranks, omega=omega, device=str(dev))
        obj.device = dev
        obj.ranks = ranks
        data_shape = tuple(ckpt["data_shape"])
        obj.N, obj.T = data_shape[0], data_shape[1]
        obj.spatial_shape = tuple(data_shape[2:])
        obj.n_spatial_modes = len(ranks) - 1
        obj.H = obj.spatial_shape[0]
        obj.W = obj.spatial_shape[1]
        obj.data = None

        if obj.n_spatial_modes == 2:
            obj.basis_net = Tensor_inr_3D(ranks, omega=omega).to(dev)
        else:
            obj.basis_net = Tensor_inr_4D(ranks, omega=omega).to(dev)
        obj.basis_net.load_state_dict(ckpt["basis_net"])
        obj.basis_net.eval()

        obj.core = ckpt["core"].to(dev)
        obj.core.requires_grad = False
        obj.optimizer = None

        obj.mode_idx = [
            torch.arange(n, device=dev).float() / max(n - 1, 1)
            for n in (obj.T, *obj.spatial_shape)
        ]
        obj.t_idx = obj.mode_idx[0]
        obj.y_idx = obj.mode_idx[1]
        obj.x_idx = obj.mode_idx[2]
        if obj.n_spatial_modes == 3:
            obj.z_idx = obj.mode_idx[3]

        obj.train_losses = []
        obj.eval_history = []
        obj.best_rmse = float(ckpt.get("rmse", float("inf")))
        return obj
