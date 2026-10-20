# Adapted from SDIFT (Chen et al., 2025, built on EDM by Karras et al., 2022): https://github.com/OceanSTARLab/SDIFT
# Core component of DyPSI (NeurIPS 2026).
"""Joint conditional sampler: posterior sampling over concatenated FTM cores
conditioned on an observed prefix, followed by inverse FTM decoding of the
physical field and the sensor-position field (SPF), and top-k sensor readout.
All shapes and paths come from the config dict.
"""

import numpy as np
import torch

from dypsi.data import topk_readout
from dypsi.diffusion import TemporalGPNoise, conditional_heun_sample
from dypsi.models.unet import Spatial_temporal_UNet
from dypsi.models.wrappers import FoldSpatialModeUNet
from dypsi.training.ftm_trainer import FTMTrainer
from dypsi.utils import get_device

__all__ = ["JointConditionalSampler"]


class JointConditionalSampler:
    """Conditional posterior sampler + joint field/trajectory decoder.

    Args:
        denoiser_ckpt: Checkpoint of the trained joint denoiser. Either a raw
            (EMA) UNet state dict or a dict containing an ``"ema"`` /
            ``"model"`` entry.
        norm_stats_path: ``.npz`` with ``mean`` / ``std`` of shape
            (1, 1, C, 1, 1) as written by ``train_joint_diffusion``.
        cfg: Config dict with grid/rank/FTM-checkpoint/UNet/EDM settings
            (see module docstring and ``train_joint_diffusion``).
        device: Device preference, see ``dypsi.utils.get_device``.
    """

    def __init__(self, denoiser_ckpt: str, norm_stats_path: str, cfg: dict,
                 device: str = "auto"):
        self.cfg = cfg
        self.device = get_device(device)

        self.grid_shape = tuple(cfg["grid_shape"])          # (H, W)
        self.n_sensors = int(cfg["n_sensors"])
        self.seg_len = int(cfg["seg_len"])
        self.k_obs = int(cfg["k_obs"])
        self.ranks_phy = tuple(cfg["ranks_phy"])            # (R1_phy, *spatial)
        self.ranks_sen = tuple(cfg["ranks_sen"])            # (R1_sen, *spatial)
        if len(self.ranks_phy) != len(self.ranks_sen):
            raise ValueError(
                f"ranks_phy and ranks_sen must have the same number of modes: "
                f"{self.ranks_phy} vs {self.ranks_sen}"
            )
        self.n_spatial_modes = len(self.ranks_phy) - 1      # 2 or 3
        if self.ranks_phy[1:] != self.ranks_sen[1:]:
            raise ValueError(
                f"Spatial ranks must match for joint stacking: {self.ranks_phy} vs {self.ranks_sen}"
            )
        self.n_channels = self.ranks_phy[0] + self.ranks_sen[0]  # C = R1_phy + R1_sen

        # EDM hyperparameters.
        edm_cfg = cfg.get("edm", {})
        self.sigma_min = float(edm_cfg.get("sigma_min", 0.002))
        self.sigma_max = float(edm_cfg.get("sigma_max", 80.0))
        self.rho = float(edm_cfg.get("rho", 7.0))
        self.sigma_data = float(edm_cfg.get("sigma_data", 0.5))
        self.num_steps = int(edm_cfg.get("num_steps", 50))
        self.zeta = float(cfg.get("zeta", 1.0))

        # ------------------------- denoiser -------------------------- #
        self.model = self._build_unet(cfg)
        self._load_denoiser(denoiser_ckpt)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        # --------------------- normalization stats -------------------- #
        stats = np.load(norm_stats_path)
        self.data_mean = torch.as_tensor(stats["mean"], dtype=torch.float32).to(self.device)
        self.data_std = torch.as_tensor(stats["std"], dtype=torch.float32).to(self.device) + 1e-8

        # ------------------------- FTM bases -------------------------- #
        self.ftm_phy = FTMTrainer.from_checkpoint(cfg["ftm_ckpt_phy"], device=str(self.device))
        self.ftm_sen = FTMTrainer.from_checkpoint(cfg["ftm_ckpt_sen"], device=str(self.device))
        for name, ftm, ranks in (("phy", self.ftm_phy, self.ranks_phy),
                                 ("sen", self.ftm_sen, self.ranks_sen)):
            if tuple(ftm.ranks) != ranks:
                print(f"[JointConditionalSampler] warning: cfg ranks_{name}={ranks} "
                      f"differ from checkpoint ranks={ftm.ranks}; using checkpoint.")

        # ---------------------- history encoder ----------------------- #
        hist_cfg = cfg.get("history", {})
        self.history_encoder = None
        if hist_cfg.get("enabled", False):
            from dypsi.models.history_encoder import HistoryEncoder

            self.history_encoder = HistoryEncoder(
                coord_dim=len(self.grid_shape),  # one coordinate per grid axis
                hidden_dim=int(hist_cfg.get("hidden_dim", 256)),
                num_layers=int(hist_cfg.get("num_layers", 4)),
                num_heads=int(hist_cfg.get("num_heads", 4)),
                max_len=int(hist_cfg.get("max_len", 64)),
            ).to(self.device)
            if hist_cfg.get("ckpt"):
                sd = torch.load(hist_cfg["ckpt"], map_location=self.device, weights_only=False)
                self.history_encoder.load_state_dict(sd)
            self.history_encoder.eval()

    # ------------------------------------------------------------------ #
    # Model construction / loading
    # ------------------------------------------------------------------ #
    def _build_unet(self, cfg: dict) -> torch.nn.Module:
        u = dict(cfg.get("unet", {}))
        hist = cfg.get("history", {})
        label_dim = int(u.get("label_dim") or
                        (hist.get("hidden_dim", 256) if hist.get("enabled", False) else 0))
        model = Spatial_temporal_UNet(
            img_resolution=int(u["img_resolution"]),
            in_channels=int(u.get("in_channels", self.n_channels)),
            out_channels=int(u.get("out_channels", self.n_channels)),
            label_dim=label_dim,
            model_channels=int(u.get("model_channels", 128)),
            channel_mult=list(u.get("channel_mult", [1, 2, 4, 4])),
            num_blocks=int(u.get("num_blocks", 3)),
            num_temporal_latent=int(u.get("num_temporal_latent", 16)),
            attn_resolutions=list(u.get("attn_resolutions", [12, 6])),
            dropout=float(u.get("dropout", 0.0)),  # no dropout at sampling time
            embedding_type=u.get("embedding_type", "positional"),
            encoder_type=u.get("encoder_type", "standard"),
            decoder_type=u.get("decoder_type", "standard"),
            resample_filter=list(u.get("resample_filter", [1, 1])),
        )
        # Fold the third spatial rank mode into the channel axis for
        # 3-spatial-mode latents.
        fold = bool(u.get("fold_spatial_modes", self.n_spatial_modes == 3))
        if fold:
            model = FoldSpatialModeUNet(model, n_spatial_modes=3)
        return model.to(self.device)

    def _load_denoiser(self, ckpt_path: str) -> None:
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        if isinstance(ckpt, dict) and "ema" in ckpt and isinstance(ckpt["ema"], dict):
            state = ckpt["ema"]
        elif isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
            state = ckpt["model"]
        else:
            state = ckpt  # raw UNet (EMA) state dict, as saved by training
        try:
            self.model.load_state_dict(state)
        except RuntimeError:
            # Tolerate a mismatch between a FoldSpatialModeUNet-wrapped model
            # and a checkpoint of the unwrapped base U-Net (or vice versa).
            base = getattr(self.model, "base_unet", None)
            if base is not None and not any(k.startswith("base_unet.") for k in state):
                base.load_state_dict(state)
            elif base is None and any(k.startswith("base_unet.") for k in state):
                state = {k[len("base_unet."):]: v for k, v in state.items()}
                self.model.load_state_dict(state)
            else:
                raise
        print(f"[JointConditionalSampler] loaded denoiser: {ckpt_path}")

    # ------------------------------------------------------------------ #
    # Normalization
    # ------------------------------------------------------------------ #
    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize joint cores with the training statistics: (x-mean)/std."""
        return (x - self.data_mean) / self.data_std

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Invert :meth:`normalize`: x*std + mean."""
        return x * self.data_std + self.data_mean

    # ------------------------------------------------------------------ #
    # Denoiser wrapper (EDM preconditioning)
    # ------------------------------------------------------------------ #
    def denoise_fn(self, x: torch.Tensor, sigma, t_grid: torch.Tensor,
                   history_emb: "torch.Tensor | None" = None) -> torch.Tensor:
        """EDM-preconditioned denoiser D(x; sigma, t).

        Args:
            x: Noisy joint cores, (B, T, C, *spatial_ranks).
            sigma: Noise level, scalar or (B,).
            t_grid: Per-frame time coordinates, (T,), (B, T, 1) or broadcastable.
            history_emb: Optional (B, hidden_dim) history embedding.

        Returns:
            Denoised estimate with the same shape as ``x``.
        """
        B, T = x.shape[0], x.shape[1]
        sigma = torch.as_tensor(sigma, dtype=x.dtype, device=x.device)
        if sigma.dim() == 0:
            sigma = sigma * torch.ones(B, dtype=x.dtype, device=x.device)
        sigma = torch.clamp(sigma, min=self.sigma_min)

        # EDM preconditioning coefficients.
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1.0 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4

        def bcast(c):  # (B,) -> broadcastable over x
            return c.view(-1, *([1] * (x.ndim - 1)))

        x_in = bcast(c_in) * x
        t_noise = c_noise.view(-1, 1, 1).expand(B, T, 1)  # (B, T, 1)

        if t_grid.dim() == 1:
            t_grid = t_grid.view(1, -1, 1).expand(B, T, 1)
        t_grid = t_grid.to(device=x.device, dtype=x.dtype)

        model_out = self.model(x_in, t_noise, t_grid, history_emb)
        return bcast(c_skip) * x + bcast(c_out) * model_out

    def _encode_history(self, history: "dict | None") -> "torch.Tensor | None":
        """Encode optional history dict {'coords','values','t'} -> (B, hidden_dim)."""
        if history is None or self.history_encoder is None:
            return None
        coords = torch.as_tensor(history["coords"], dtype=torch.float32, device=self.device)
        values = torch.as_tensor(history["values"], dtype=torch.float32, device=self.device)
        t = torch.as_tensor(history["t"], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            return self.history_encoder(coords, values, t)

    def _t_grid(self) -> torch.Tensor:
        """Normalized per-frame time coordinates, shape (seg_len,)."""
        return torch.linspace(0.0, 1.0, self.seg_len, device=self.device)

    def _gp_noise(self, t_grid: torch.Tensor) -> TemporalGPNoise:
        """Matern-3/2 GP-correlated temporal noise from ``cfg['gp']``."""
        gp_cfg = self.cfg.get("gp", {})
        return TemporalGPNoise(
            t_grid,
            kernel=gp_cfg.get("kernel", "matern32"),
            length_scale=float(gp_cfg.get("length_scale", 4.0)),
            signal_var=float(gp_cfg.get("signal_var", 1.0)),
            gp_gamma=float(gp_cfg.get("gp_gamma", 50.0)),
        )

    # ------------------------------------------------------------------ #
    # Sampling
    # ------------------------------------------------------------------ #
    def sample(self, observed_cores: torch.Tensor, num_steps: "int | None" = None,
               anchor: str = "linear", history: "dict | None" = None) -> torch.Tensor:
        """Posterior-sample a full segment conditioned on an observed prefix.

        Args:
            observed_cores: Normalized prefix of shape (B, k_obs, C, *spatial).
            num_steps: Number of Heun steps; defaults to ``cfg['edm']['num_steps']``.
            anchor: Prefix anchoring mode passed to ``conditional_heun_sample``.
            history: Optional history dict, see :meth:`_encode_history`.

        Returns:
            Sampled segment of shape (B, seg_len, C, *spatial), denormalized.
        """
        observed_cores = observed_cores.to(self.device)
        B = observed_cores.shape[0]
        num_steps = int(num_steps or self.num_steps)

        t_grid = self._t_grid()
        gp_noise = self._gp_noise(t_grid)
        history_emb = self._encode_history(history)

        shape = (B, self.seg_len, self.n_channels, *self.ranks_phy[1:])
        sampled = conditional_heun_sample(
            self.denoise_fn,
            shape,
            t_grid,
            num_steps=num_steps,
            sigma_min=self.sigma_min,
            sigma_max=self.sigma_max,
            rho=self.rho,
            observed=observed_cores,
            gp_noise=gp_noise,
            history_emb=history_emb,
            anchor=anchor,
            zeta=self.zeta,
            device=self.device,
        )
        return self.denormalize(sampled.float())

    # ------------------------------------------------------------------ #
    # Decoding
    # ------------------------------------------------------------------ #
    def decode_frame(self, joint_frame: torch.Tensor, frame_idx: int,
                     seg_len: int) -> dict:
        """Decode one joint core frame into the physical field and the SPF.

        Args:
            joint_frame: Denormalized joint core of shape (C, *spatial_ranks)
                (extra leading singleton dims are squeezed).
            frame_idx: Temporal index of the frame within the segment.
            seg_len: Segment length; the time coordinate is
                ``frame_idx / (seg_len - 1)`` in [0, 1].

        Returns:
            Dict with ``'field'`` (*grid_shape) physical field, ``'spf'``
            (*grid_shape) sensor-position field, and ``'sensor_coords'``
            (k, D) grid coordinates from top-k readout of the SPF
            (D = 2 or 3, matching ``grid_shape``).
        """
        joint_frame = joint_frame.to(self.device).float()
        target_dim = 1 + self.n_spatial_modes
        while joint_frame.dim() > target_dim:
            joint_frame = joint_frame.squeeze(0)
        spatial_ranks = self.ranks_phy[1:]
        C = joint_frame.shape[0]
        r1p, r1s = self.ranks_phy[0], self.ranks_sen[0]
        if C != r1p + r1s:
            raise ValueError(f"Expected {r1p + r1s} channels, got {C}")
        if tuple(joint_frame.shape[1:]) != tuple(spatial_ranks):
            raise ValueError(
                f"Expected spatial ranks {spatial_ranks}, got "
                f"{tuple(joint_frame.shape[1:])}"
            )

        phy_flat = joint_frame[:r1p].reshape(-1)        # (R1_phy * prod(spatial),)
        sen_flat = joint_frame[r1p:r1p + r1s].reshape(-1)

        # Coordinate grid over the dense output grid, order (t, y, x[, z]).
        axes = [torch.linspace(0, 1, n, device=self.device) for n in self.grid_shape]
        mesh = torch.meshgrid(*axes, indexing="ij")
        n_pts = 1
        for n in self.grid_shape:
            n_pts *= n
        t_val = frame_idx / max(seg_len - 1, 1)
        t_coords = torch.full((n_pts, 1), t_val, device=self.device)
        full_coords = torch.cat(
            [t_coords] + [m.reshape(-1, 1) for m in mesh], dim=-1
        )  # (n_pts, 1 + n_spatial)

        with torch.no_grad():
            self.ftm_phy.basis_net.mode = "sampling"
            basis_phy = self.ftm_phy.basis_net(input_ind_sampl=full_coords)  # (n_pts, R1p*prod(spatial))
            field = (basis_phy @ phy_flat).view(*self.grid_shape)

            self.ftm_sen.basis_net.mode = "sampling"
            basis_sen = self.ftm_sen.basis_net(input_ind_sampl=full_coords)  # (n_pts, R1s*prod(spatial))
            spf = (basis_sen @ sen_flat).view(*self.grid_shape)

        sensor_coords = topk_readout(
            spf.detach().cpu().numpy(),
            self.n_sensors,
            use_nms=bool(self.cfg.get("use_nms", True)),
            nms_size=int(self.cfg.get("nms_size", 3)),
        )
        return {"field": field, "spf": spf, "sensor_coords": sensor_coords}

    # ------------------------------------------------------------------ #
    # Autoregressive long-sequence generation
    # ------------------------------------------------------------------ #
    def generate_sequence(self, init_prefix_cores: torch.Tensor, total_len: int) -> dict:
        """Autoregressively generate a long sequence with a sliding window.

        Args:
            init_prefix_cores: Raw (unnormalized) prefix of shape
                (B, k_obs, C, R2, R3).
            total_len: Total number of frames to generate.

        Returns:
            Dict with ``'cores'`` (B, total_len, C, *spatial_ranks),
            ``'fields'`` (B, total_len, *grid_shape), ``'spfs'`` (B,
            total_len, *grid_shape) and ``'trajectories'`` (B, total_len, k,
            D) with ``D = len(grid_shape)``.
        """
        from dypsi.diffusion import SlidingWindowGenerator

        init_prefix = self.normalize(init_prefix_cores.to(self.device).float())
        t_grid = self._t_grid()
        gp_noise = self._gp_noise(t_grid)
        sample_kwargs = dict(
            t_grid=t_grid,
            num_steps=self.num_steps,
            sigma_min=self.sigma_min,
            sigma_max=self.sigma_max,
            rho=self.rho,
            gp_noise=gp_noise,
            anchor="linear",
            zeta=self.zeta,
            device=self.device,
        )
        generator = SlidingWindowGenerator(
            self.denoise_fn, self.seg_len, self.k_obs, sample_kwargs
        )
        cores_norm = generator.generate(init_prefix, total_len)
        cores = self.denormalize(cores_norm.float())

        B, T_out = cores.shape[0], cores.shape[1]
        fields = torch.zeros(B, T_out, *self.grid_shape)
        spfs = torch.zeros(B, T_out, *self.grid_shape)
        trajs = np.zeros((B, T_out, self.n_sensors, len(self.grid_shape)),
                         dtype=np.float32)
        for b in range(B):
            for t in range(T_out):
                dec = self.decode_frame(cores[b, t], t, total_len)
                fields[b, t] = dec["field"].cpu()
                spfs[b, t] = dec["spf"].cpu()
                coords = dec["sensor_coords"]
                if isinstance(coords, torch.Tensor):
                    coords = coords.cpu().numpy()
                trajs[b, t] = np.asarray(coords, dtype=np.float32)[: self.n_sensors]

        return {"cores": cores, "fields": fields, "spfs": spfs, "trajectories": trajs}
