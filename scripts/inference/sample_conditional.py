# Core component of DyPSI (NeurIPS 2026).
"""Conditional sampling with the trained DyPSI joint diffusion prior.

Given observed prefix cores (K_obs frames), imputes the unobserved
remainder of the segment and decodes each sampled joint frame into a
physical field, a sensor placement field (SPF), and sensor coordinates.
"""

import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np

from dypsi.inference.joint_sampler import JointConditionalSampler
from dypsi.utils import get_device, load_config, set_seed

logger = logging.getLogger("dypsi.sample_conditional")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run DyPSI conditional sampling: impute unobserved "
                    "segment frames from observed prefix cores.")
    parser.add_argument("--config", required=True,
                        help="Path to the dataset YAML config.")
    parser.add_argument("--denoiser_ckpt", required=True,
                        help="Path to the trained joint diffusion denoiser "
                             "checkpoint.")
    parser.add_argument("--norm_stats", required=True,
                        help="Path to the core normalization statistics used "
                             "during diffusion training.")
    parser.add_argument("--ftm_ckpt_phy", required=True,
                        help="Path to the phy-modality FTM checkpoint "
                             "(written by scripts/train/train_ftm.py).")
    parser.add_argument("--ftm_ckpt_sen", required=True,
                        help="Path to the sen-modality (SPF) FTM checkpoint "
                             "(written by scripts/train/train_ftm.py).")
    parser.add_argument("--observed_cores", required=True,
                        help="Observed prefix cores as .npy, shaped "
                             "(K_obs, C, R2, R3) or a batch thereof.")
    parser.add_argument("--outdir", required=True,
                        help="Directory for the decoded predictions (.npz).")
    parser.add_argument("--num_steps", type=int, default=None,
                        help="Sampler steps. Defaults to sampling.num_steps.")
    parser.add_argument("--anchor", default=None,
                        help="Prefix anchoring scheme. Defaults to "
                             "sampling.anchor.")
    parser.add_argument("--visualize", action="store_true",
                        help="Optionally render decoded fields to PNG "
                             "(matplotlib Agg backend).")
    parser.add_argument("--device", default="auto",
                        help="Device preference: 'auto', 'cpu', 'cuda', ...")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed.")
    return parser.parse_args()


def visualize(fields, spfs, sensor_coords, outdir, n_frames=4):
    """Render a few decoded frames to PNG using the Agg backend."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vis_dir = os.path.join(outdir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    indices = np.linspace(0, fields.shape[0] - 1,
                          min(n_frames, fields.shape[0])).astype(int)
    for frame_idx in indices:
        field = np.asarray(fields[frame_idx])
        if field.ndim == 3:      # (C, H, W) -> first channel
            field = field[0]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].imshow(field, cmap="viridis")
        axes[0].set_title(f"Predicted field (frame {frame_idx})")
        spf = np.asarray(spfs[frame_idx])
        if spf.ndim > 2:
            spf = spf.reshape(-1, *spf.shape[-2:]).max(axis=0)
        axes[1].imshow(spf, cmap="magma")
        coords = np.asarray(sensor_coords[frame_idx])
        axes[1].scatter(coords[:, 1], coords[:, 0], c="cyan", s=10)
        axes[1].set_title("SPF + sensor positions")
        fig.tight_layout()
        fig.savefig(os.path.join(vis_dir, f"frame_{frame_idx:03d}.png"),
                    dpi=150)
        plt.close(fig)
    logger.info("Visualizations written to %s", vis_dir)


def build_sampler_cfg(cfg, args):
    """Assemble the flat config dict read by ``JointConditionalSampler``.

    Maps the YAML sections (``dataset`` / ``joint`` / ``ftm`` / ``diffusion`` /
    ``sampling``) onto flat keys and injects the FTM checkpoint paths from
    the command line.
    """
    ds_cfg = cfg["dataset"]
    joint_cfg = cfg["joint"]
    ftm_cfg = cfg["ftm"]
    diff_cfg = cfg.get("diffusion", {})
    sampling_cfg = cfg.get("sampling", {})

    ranks_phy = list(ftm_cfg["phy"]["ranks"])
    ranks_sen = list(ftm_cfg["sen"]["ranks"])
    if ranks_phy[1:] != ranks_sen[1:]:
        raise ValueError(
            f"Spatial FTM ranks must match for joint stacking: "
            f"phy {ranks_phy} vs sen {ranks_sen}")

    unet_cfg = dict(diff_cfg.get("unet", {}))
    unet_cfg.setdefault("img_resolution", max(ranks_phy[1], ranks_phy[2]))
    in_channels = ranks_phy[0] + ranks_sen[0]
    if len(ranks_phy) == 4:
        # 3-spatial-mode cores (car aerodynamics) fold R4 into the channel axis.
        unet_cfg["fold_spatial_modes"] = True
        unet_cfg["img_resolution"] = ranks_phy[1]  # R2 (= R3 for the car grid)
        in_channels *= ranks_phy[3]
    unet_cfg["in_channels"] = in_channels

    edm_cfg = dict(diff_cfg.get("edm", {}))
    edm_cfg["num_steps"] = sampling_cfg.get("num_steps", 50)

    return {
        "grid_shape": ds_cfg["grid_shape"],
        "n_sensors": int(cfg["trajectory_optimization"]["n_sensors"]),
        "seg_len": int(joint_cfg["seg_len"]),
        "k_obs": int(joint_cfg["k_obs"]),
        "ranks_phy": ranks_phy,
        "ranks_sen": ranks_sen,
        "spf_bandwidth": joint_cfg.get("spf_bandwidth"),
        "ftm_ckpt_phy": args.ftm_ckpt_phy,
        "ftm_ckpt_sen": args.ftm_ckpt_sen,
        "unet": unet_cfg,
        "edm": edm_cfg,
        "gp": diff_cfg.get("gp", {}),
        "zeta": sampling_cfg.get("zeta", 1.0),
        "use_nms": sampling_cfg.get("use_nms", True),
        "nms_size": sampling_cfg.get("nms_size", 3),
        "history": diff_cfg.get("history", {"enabled": False}),
    }


def _to_numpy(x):
    """Convert a torch tensor or array-like to a NumPy array."""
    import torch

    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    cfg = load_config(args.config)
    set_seed(args.seed)
    device = get_device(args.device)
    os.makedirs(args.outdir, exist_ok=True)

    sampling_cfg = cfg.get("sampling", {})
    num_steps = args.num_steps or sampling_cfg.get("num_steps", 50)
    anchor = args.anchor or sampling_cfg.get("anchor", "linear")
    seg_len = cfg["joint"]["seg_len"]

    observed_cores = np.load(args.observed_cores)
    logger.info("Observed prefix cores %s from %s",
                observed_cores.shape, args.observed_cores)

    sampler_cfg = build_sampler_cfg(cfg, args)
    sampler = JointConditionalSampler(
        args.denoiser_ckpt, args.norm_stats, sampler_cfg, device=str(device))

    # The sampler expects a normalized, batched prefix (B, K_obs, C, R2, R3).
    import torch

    observed = torch.as_tensor(observed_cores, dtype=torch.float32)
    if observed.dim() == 4:
        observed = observed.unsqueeze(0)
    observed = sampler.normalize(observed)

    logger.info("Sampling segment (num_steps=%d, anchor=%s)",
                num_steps, anchor)
    joint_frames = sampler.sample(observed, num_steps=num_steps,
                                  anchor=anchor)  # (B, seg_len, C, R2, R3)

    # Decode every frame of the sampled joint sequence (first batch entry).
    fields, spfs, sensor_coords = [], [], []
    n_decode = min(seg_len, joint_frames.shape[1])
    for frame_idx in range(n_decode):
        decoded = sampler.decode_frame(joint_frames[0, frame_idx], frame_idx,
                                       seg_len)
        fields.append(_to_numpy(decoded["field"]))
        spfs.append(_to_numpy(decoded["spf"]))
        sensor_coords.append(_to_numpy(decoded["sensor_coords"]))
    fields = np.stack(fields)
    spfs = np.stack(spfs)
    sensor_coords = np.stack(sensor_coords)

    out_path = os.path.join(args.outdir, "conditional_samples.npz")
    np.savez_compressed(
        out_path,
        fields=fields,
        spfs=spfs,
        sensor_coords=sensor_coords,
        observed_cores=observed_cores,
        meta=json.dumps({
            "dataset": cfg["dataset"]["name"],
            "num_steps": num_steps,
            "anchor": anchor,
            "seg_len": seg_len,
        }),
    )

    k_obs = cfg["joint"]["k_obs"]
    print("=== DyPSI conditional sampling summary ===")
    print(f"dataset          : {cfg['dataset']['name']}")
    print(f"observed prefix  : {observed_cores.shape} (K_obs={k_obs})")
    print(f"decoded frames   : {fields.shape[0]} / seg_len={seg_len}")
    print(f"field shape      : {fields.shape}")
    print(f"spf shape        : {spfs.shape}")
    print(f"sensor coords    : {sensor_coords.shape}")
    print(f"output           : {out_path}")

    if args.visualize:
        visualize(fields, spfs, sensor_coords, args.outdir)


if __name__ == "__main__":
    main()
