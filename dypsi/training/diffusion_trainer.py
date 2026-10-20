# Adapted from SDIFT (Chen et al., 2025, built on EDM by Karras et al., 2022): https://github.com/OceanSTARLab/SDIFT
# Core component of DyPSI (NeurIPS 2026).
"""Joint diffusion training over concatenated physical-field and SPF FTM cores.

Driven by a plain config dict (typically loaded from YAML) with sections
``data``, ``unet``, ``edm``, ``gp``, ``history`` and ``train``; see
``train_joint_diffusion``.
"""

import logging
import os
from dataclasses import fields

import numpy as np
import torch

from dypsi.data import JointCoreDataset
from dypsi.diffusion import EDMConfig, JointEDM, TemporalGPNoise
from dypsi.models.unet import Spatial_temporal_UNet
from dypsi.models.wrappers import FoldSpatialModeUNet
from dypsi.utils import get_device, set_seed
from torch.utils.data import DataLoader

__all__ = ["build_denoiser", "train_joint_diffusion"]

_UNET_DEFAULTS = dict(
    model_channels=128,
    channel_mult=[1, 2, 4, 4],
    num_blocks=3,
    num_temporal_latent=16,
    attn_resolutions=[12, 6],
    dropout=0.1,
    label_dim=0,
)


def build_denoiser(cfg: dict) -> torch.nn.Module:
    """Build the ``Spatial_temporal_UNet`` denoiser from ``cfg['unet']``.

    Requires keys ``img_resolution`` and ``in_channels`` (= R1_phy + R1_sen,
    or ``(R1_phy + R1_sen) * R4`` when folding); other U-Net hyperparameters
    fall back to ``_UNET_DEFAULTS``, and ``label_dim`` defaults to
    ``history.hidden_dim`` when the history encoder is enabled. With
    ``fold_spatial_modes: True`` the U-Net is wrapped in
    ``FoldSpatialModeUNet`` so it accepts 3-spatial-mode latents
    (B, T, C, R2, R3, R4) with R4 folded into the channel axis.

    Returns:
        The constructed denoiser (not moved to any device); a
        ``FoldSpatialModeUNet`` wrapper when folding is enabled.
    """
    u = dict(_UNET_DEFAULTS)
    u.update(cfg.get("unet", {}))

    hist = cfg.get("history", {})
    if hist.get("enabled", False) and not u.get("label_dim"):
        u["label_dim"] = int(hist.get("hidden_dim", 256))

    if "img_resolution" not in u or "in_channels" not in u:
        raise KeyError("cfg['unet'] must define 'img_resolution' and 'in_channels' "
                       "(= R1_phy + R1_sen, or (R1_phy + R1_sen) * R4 when folding).")

    model = Spatial_temporal_UNet(
        img_resolution=int(u["img_resolution"]),
        in_channels=int(u["in_channels"]),
        out_channels=int(u.get("out_channels", u["in_channels"])),
        label_dim=int(u["label_dim"]),
        model_channels=int(u["model_channels"]),
        channel_mult=list(u["channel_mult"]),
        num_blocks=int(u["num_blocks"]),
        num_temporal_latent=int(u["num_temporal_latent"]),
        attn_resolutions=list(u["attn_resolutions"]),
        dropout=float(u["dropout"]),
        embedding_type=u.get("embedding_type", "positional"),
        encoder_type=u.get("encoder_type", "standard"),
        decoder_type=u.get("decoder_type", "standard"),
        resample_filter=list(u.get("resample_filter", [1, 1])),
    )
    fold = bool(u.get("fold_spatial_modes", False))
    if fold:
        # Fold the third spatial rank mode into the channel axis for
        # 3D-spatial benchmarks.
        model = FoldSpatialModeUNet(model, n_spatial_modes=3)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[build_denoiser] Spatial_temporal_UNet: in_channels={u['in_channels']}, "
          f"label_dim={u['label_dim']}, fold_spatial_modes={fold}, params={n_params:,}")
    return model


def _build_edm_config(cfg: dict) -> EDMConfig:
    """Instantiate ``EDMConfig`` from ``cfg['edm']``, ignoring unknown keys."""
    valid = {f.name for f in fields(EDMConfig)}
    kwargs = {k: v for k, v in cfg.get("edm", {}).items() if k in valid}
    return EDMConfig(**kwargs)


def _history_embedding(encoder, signals: torch.Tensor, t_batch: torch.Tensor,
                       hist_cfg: dict) -> torch.Tensor:
    """Build history embeddings from the observed prefix of a training batch.

    Uniformly subsamples (frame, spatial) locations from the first ``k_obs``
    frames of each sequence and encodes (coords, values, t) triplets.

    Args:
        encoder: ``HistoryEncoder`` module.
        signals: Batch of joint cores, (B, T, C, *spatial).
        t_batch: Per-frame time coordinates, (B, T, 1).
        hist_cfg: ``cfg['history']`` sub-dict.

    Returns:
        Tensor of shape (B, hidden_dim).
    """
    B, T, C = signals.shape[:3]
    spatial = signals.shape[3:]
    k_obs = int(hist_cfg.get("k_obs", max(T // 2, 1)))
    k_obs = min(k_obs, T)
    max_len = int(hist_cfg.get("max_len", 64))

    f_idx = torch.randint(0, k_obs, (B, max_len), device=signals.device)      # (B, L)
    dim_idx = [
        torch.randint(0, n, (B, max_len), device=signals.device)              # (B, L)
        for n in spatial
    ]

    b_arange = torch.arange(B, device=signals.device).unsqueeze(1)            # (B, 1)
    # Channel-averaged core value at each sampled location: (B, L, 1).
    values = signals[(b_arange, f_idx, slice(None), *dim_idx)].mean(dim=2, keepdim=True)
    # Normalized coordinates in [0, 1]: (B, L, n_spatial), in axis order.
    coords = torch.stack(
        [d.float() / max(n - 1, 1) for d, n in zip(dim_idx, spatial)], dim=-1
    )
    t_sel = t_batch[b_arange, f_idx]                                          # (B, L, 1)
    return encoder(coords, values, t_sel)


def train_joint_diffusion(cfg: dict) -> str:
    """Run the full joint diffusion training loop.

    Builds the dataset + dataloader, denoiser, optional history encoder,
    ``JointEDM`` wrapper and AdamW optimizer; trains with gradient
    accumulation, linear LR warmup, gradient clipping and EMA; periodically
    saves checkpoints to ``cfg['outdir']/checkpoints`` and the normalization
    statistics to ``cfg['outdir']/norm_stats.npz``.

    Args:
        cfg: Config dict with sections ``data``, ``unet``, ``edm``, ``gp``,
            ``history`` and ``train``; all paths and hyperparameters come
            from this dict.

    Returns:
        Path to the checkpoint directory (``cfg['outdir']/checkpoints``).
    """
    train_cfg = cfg.get("train", {})
    seed = int(cfg.get("seed", 42))
    set_seed(seed)
    device = get_device(cfg.get("device", "auto"))

    outdir = cfg["outdir"]
    ckpt_dir = os.path.join(outdir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    logging.basicConfig(
        filename=os.path.join(outdir, "train.log"),
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("dypsi.train_joint_diffusion")
    logger.info(f"Config: {cfg}")
    print(f"[train_joint_diffusion] device={device}, outdir={outdir}")

    # ------------------------------ data ------------------------------ #
    data_cfg = cfg["data"]
    dataset = JointCoreDataset(
        phy_core_path=data_cfg["phy_core_path"],
        sen_core_path=data_cfg["sen_core_path"],
        seg_len=int(data_cfg.get("seg_len", 16)),
        stride=int(data_cfg.get("stride", 5)),
        normalize=bool(data_cfg.get("normalize", True)),
    )
    batch_size = int(train_cfg.get("batch_size", 4))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=0, drop_last=True)

    mean, std = dataset.get_norm_stats()
    np.savez(os.path.join(outdir, "norm_stats.npz"), mean=mean, std=std)
    logger.info(f"Dataset: {len(dataset)} segments, data shape {tuple(dataset.data.shape)}")
    print(f"[train_joint_diffusion] dataset {tuple(dataset.data.shape)}")

    # ------------------------------ model ----------------------------- #
    model = build_denoiser(cfg)
    edm_cfg = _build_edm_config(cfg)

    # GP-correlated temporal noise (Matern-3/2) shared across training.
    t_grid_1d = dataset.t_grid[0].reshape(-1).to(device)  # (T,)
    gp_cfg = cfg.get("gp", {})
    gp_noise = TemporalGPNoise(
        t_grid_1d,
        kernel=gp_cfg.get("kernel", "matern32"),
        length_scale=float(gp_cfg.get("length_scale", 4.0)),
        signal_var=float(gp_cfg.get("signal_var", 1.0)),
        gp_gamma=float(gp_cfg.get("gp_gamma", 50.0)),
    )

    edm = JointEDM(model, edm_cfg, batch_size=batch_size, gp_noise=gp_noise, device=device)

    # Optional history encoder for prefix conditioning.
    hist_cfg = cfg.get("history", {})
    history_encoder = None
    params = list(model.parameters())
    if hist_cfg.get("enabled", False):
        from dypsi.models.history_encoder import HistoryEncoder

        history_encoder = HistoryEncoder(
            coord_dim=len(dataset.data.shape[3:]),  # one coordinate per spatial mode
            hidden_dim=int(hist_cfg.get("hidden_dim", 256)),
            num_layers=int(hist_cfg.get("num_layers", 4)),
            num_heads=int(hist_cfg.get("num_heads", 4)),
            max_len=int(hist_cfg.get("max_len", 64)),
        ).to(device)
        params += list(history_encoder.parameters())

    lr = float(train_cfg.get("lr", 1e-4))
    optimizer = torch.optim.AdamW(params, lr=lr)

    # ------------------------------ loop ------------------------------ #
    num_steps = int(train_cfg.get("num_steps", 60000))
    accumulation_steps = int(train_cfg.get("accumulation_steps", 1))
    warmup = int(train_cfg.get("warmup", 1000))
    grad_clip = float(train_cfg.get("grad_clip", 1.0))
    save_model_iters = int(train_cfg.get("save_model_iters", 5000))
    log_step = int(train_cfg.get("log_step", 40))

    try:
        from tqdm import tqdm
        progress = tqdm(total=num_steps, desc="Training Joint Diffusion")
    except ImportError:
        progress = None

    logger.info("Training started")
    train_loss = 0.0
    data_iterator = iter(loader)

    for step in range(num_steps):
        optimizer.zero_grad()
        batch_loss = torch.zeros((), device=device)

        for _ in range(accumulation_steps):
            try:
                signal_batch, t_batch = next(data_iterator)
            except StopIteration:
                data_iterator = iter(loader)
                signal_batch, t_batch = next(data_iterator)

            signal_batch = signal_batch.to(device)
            t_batch = t_batch.to(device)

            history_emb = None
            if history_encoder is not None:
                history_emb = _history_embedding(history_encoder, signal_batch, t_batch, hist_cfg)

            loss = edm.train_step(signal_batch, t_batch, history_emb=history_emb)
            loss = loss / accumulation_steps
            loss.backward()
            batch_loss += loss.detach()

        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(params, grad_clip)

        # Linear LR warmup.
        current_lr = lr * min(step / max(warmup, 1), 1.0)
        for g in optimizer.param_groups:
            g["lr"] = current_lr
        optimizer.step()

        edm.update_ema(step)
        train_loss += batch_loss.item()

        if step % log_step == 0 or step == num_steps - 1:
            avg_loss = train_loss / (step + 1)
            logger.info(f"Step {step:06d}, Loss: {avg_loss:.6f}, LR: {current_lr:.6e}")
            if progress is not None:
                progress.set_postfix({"loss": f"{avg_loss:.4f}", "lr": f"{current_lr:.2e}"})

        if (step > 0) and (step % save_model_iters == 0 or step == num_steps - 1):
            ckpt_path = os.path.join(ckpt_dir, f"ema_{step}.pth")
            torch.save(edm.state_dict(), ckpt_path)
            logger.info(f"Saved checkpoint: {ckpt_path}")
            print(f"[train_joint_diffusion] saved {ckpt_path}")

        if progress is not None:
            progress.update(1)

    if progress is not None:
        progress.close()

    final_path = os.path.join(ckpt_dir, "ema_final.pth")
    torch.save(edm.state_dict(), final_path)
    logger.info("Training finished")
    print(f"[train_joint_diffusion] done. Final avg loss: {train_loss / num_steps:.6f}")
    print(f"[train_joint_diffusion] checkpoints: {ckpt_dir}")
    print(f"[train_joint_diffusion] norm stats: {os.path.join(outdir, 'norm_stats.npz')}")
    return ckpt_dir
