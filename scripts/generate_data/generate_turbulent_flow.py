# Core component of DyPSI (NeurIPS 2026).
"""Generate DyPSI training data for the turbulent_flow dataset.

Splits raw field snapshots into sliding-window segments and optimizes a
differentiable sensor trajectory per segment against a teacher
reconstructor, saving the trajectories and their Sensor Placement Field
(SPF) rasterizations for FTM and joint-diffusion training.
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np

from dypsi.data import (
    TrajectoryOptimizer,
    build_spf_sequence,
    load_turbulent_flow,
)
from dypsi.utils import get_device, load_config, set_seed
from scripts.generate_data.teacher_adapters import build_teacher_reconstructor

logger = logging.getLogger("dypsi.generate_turbulent_flow")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate optimized sensor trajectories and SPF fields "
                    "for the turbulent_flow dataset.")
    parser.add_argument("--config", required=True,
                        help="Path to the dataset YAML config.")
    parser.add_argument("--outdir", required=True,
                        help="Output directory for trajectories and SPF fields.")
    parser.add_argument("--num_sensors", type=int, default=None,
                        help="Sensor budget k. Defaults to the first entry of "
                             "dataset.sensor_budgets in the config.")
    parser.add_argument("--n_iterations", type=int, default=None,
                        help="Trajectory optimization iterations per segment. "
                             "Defaults to trajectory_optimization.n_iterations.")
    parser.add_argument("--physense_repo", default=None,
                        help="Path to a local clone of the PhySense repository.")
    parser.add_argument("--device", default="auto",
                        help="Device preference: 'auto', 'cpu', 'cuda', ...")
    parser.add_argument("--max_segments", type=int, default=None,
                        help="Optional cap on the number of segments (debugging).")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed.")
    return parser.parse_args()


def random_init_positions(n_sensors, spatial_dims, grid_shape, device, seed):
    """Uniform random initial sensor positions inside the grid domain."""
    import torch

    generator = torch.Generator(device="cpu").manual_seed(seed)
    lows = torch.zeros(spatial_dims)
    highs = torch.tensor([float(s - 1) for s in grid_shape])
    positions = lows + (highs - lows) * torch.rand(
        n_sensors, spatial_dims, generator=generator)
    return positions.to(device)


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    cfg = load_config(args.config)
    set_seed(args.seed)
    device = get_device(args.device)
    logger.info("Loaded config from %s (device=%s)", args.config, device)

    ds_cfg = cfg["dataset"]
    opt_cfg = cfg["trajectory_optimization"]
    joint_cfg = cfg["joint"]

    num_sensors = args.num_sensors or ds_cfg["sensor_budgets"][0]
    n_iterations = args.n_iterations or opt_cfg["n_iterations"]
    n_frames = opt_cfg["n_frames"]
    stride = joint_cfg["stride"]
    grid_shape = tuple(ds_cfg["grid_shape"])
    bandwidth = joint_cfg["spf_bandwidth"]

    # 1) Load the raw field sequence, (T, C, *grid).
    fields = load_turbulent_flow(ds_cfg["raw_path"],
                                 num_frames=ds_cfg.get("num_frames"))
    if fields.ndim == 3:
        # (T, H, W) -> (T, H, W, 1): TrajectoryOptimizer expects a channel axis.
        fields = fields[..., None]
    total_frames = fields.shape[0]
    logger.info("Loaded %d frames from %s", total_frames, ds_cfg["raw_path"])

    # 2) Build the teacher reconstructor (external PhySense dependency).
    reconstructor = build_teacher_reconstructor(
        opt_cfg.get("teacher", "physense"),
        physense_repo=args.physense_repo,
        device=str(device),
        grid_shape=grid_shape,
        n_channels=ds_cfg["n_channels"],
    )

    # 3) Sliding-window segmentation + per-segment trajectory optimization.
    traj_dir = os.path.join(args.outdir, "trajectories")
    spf_dir = os.path.join(args.outdir, "spf")
    os.makedirs(traj_dir, exist_ok=True)
    os.makedirs(spf_dir, exist_ok=True)

    starts = list(range(0, total_frames - n_frames + 1, stride))
    if args.max_segments is not None:
        starts = starts[: args.max_segments]
    logger.info("Optimizing %d segments (n_frames=%d, stride=%d, k=%d)",
                len(starts), n_frames, stride, num_sensors)

    spf_chunks = []
    for seg_idx, start in enumerate(starts):
        segment = fields[start:start + n_frames]
        init_positions = random_init_positions(
            num_sensors, ds_cfg["spatial_dims"], grid_shape, device,
            seed=args.seed + seg_idx)
        optimizer = TrajectoryOptimizer(
            reconstructor=reconstructor,
            n_sensors=num_sensors,
            n_frames=n_frames,
            lambda_gp=opt_cfg["lambda_gp"],
            gamma_dyn=opt_cfg["gamma_dyn"],
            v_max=opt_cfg["v_max"],
            a_max=opt_cfg["a_max"],
            gp_length_scale=opt_cfg["gp_length_scale"],
            gp_sigma=opt_cfg["gp_sigma"],
            n_recon_samples=opt_cfg["n_recon_samples"],
            euler_steps=opt_cfg["euler_steps"],
            init_positions=init_positions,
            use_gradient_checkpointing=opt_cfg["use_gradient_checkpointing"],
            device=str(device),
        )
        trajectory, history = optimizer.optimize(
            segment, n_iterations=n_iterations, lr=opt_cfg["lr"],
            domain_constraint=None)

        # 4) Save the segment trajectory (.pth) and its SPF field (.npy).
        import torch

        traj_path = os.path.join(traj_dir, f"segment_{seg_idx:04d}.pth")
        torch.save({
            "trajectory": trajectory,          # (T, k, D)
            "segment_start": start,
            "config": {
                "dataset": ds_cfg["name"],
                "n_sensors": num_sensors,
                "n_frames": n_frames,
                "trajectory_optimization": opt_cfg,
            },
            "history": history,
        }, traj_path)

        traj_batch = trajectory.detach().cpu().numpy()[None]  # (1, T, k, D)
        spf = build_spf_sequence(traj_batch, grid_shape, bandwidth)
        spf = np.asarray(spf, dtype=np.float32)        # (1, T, *grid)
        np.save(os.path.join(spf_dir, f"segment_{seg_idx:04d}.npy"), spf[0])
        spf_chunks.append(spf)
        final_loss = float(history[-1]["total"]) if history else float("nan")
        logger.info("Segment %d/%d done: trajectory -> %s (final loss %.6f)",
                    seg_idx + 1, len(starts), traj_path, final_loss)

    # Combined SPF tensor (N, T, *grid) for sen-modality FTM training.
    if spf_chunks:
        combined = np.concatenate(spf_chunks, axis=0)
        combined_path = os.path.join(args.outdir, "spf_fields.npy")
        np.save(combined_path, combined)
        logger.info("Saved combined SPF fields %s -> %s",
                    combined.shape, combined_path)

    logger.info("All outputs written to %s", args.outdir)


if __name__ == "__main__":
    main()
