# Core component of DyPSI (NeurIPS 2026).
"""Generate DyPSI training data for the car_aerodynamics dataset.

Splits the surface-pressure sequence into sliding-window segments and
optimizes a differentiable sensor trajectory per segment against a teacher
reconstructor, snapping sensors onto the car surface vertices.
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
    load_car_aerodynamics,
)
from dypsi.utils import get_device, load_config, set_seed
from scripts.generate_data.teacher_adapters import build_teacher_reconstructor

logger = logging.getLogger("dypsi.generate_car_aerodynamics")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate optimized sensor trajectories and SPF fields "
                    "for the car_aerodynamics dataset.")
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


def make_surface_constraint(n_vertices, device):
    """Build a domain constraint snapping sensor positions to mesh vertices.

    The mesh is treated as a degenerate grid (V, 1) over flattened vertex
    indices; the index is rounded to the nearest vertex and clamped into
    range with straight-through gradients. Returns a callable mapping
    positions (k, 2) to snapped (k, 2) positions.
    """
    import torch

    def project(positions):
        idx = positions[..., 0].round().clamp(0.0, float(n_vertices - 1))
        snapped = torch.stack([idx, torch.zeros_like(idx)], dim=-1)
        return positions + (snapped - positions).detach()

    return project


def random_init_positions(n_vertices, n_sensors, device, seed):
    """Sample initial sensor positions uniformly over the vertex indices."""
    import torch

    generator = torch.Generator(device="cpu").manual_seed(seed)
    idx = torch.randint(0, n_vertices, (n_sensors,), generator=generator)
    positions = torch.stack(
        [idx.float(), torch.zeros(n_sensors)], dim=-1)
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
    bandwidth = joint_cfg["spf_bandwidth"]

    # 1) Load the raw surface-pressure sequence and the mesh.
    fields, mesh_info = load_car_aerodynamics(ds_cfg["raw_path"])
    total_frames = fields.shape[0]
    vertices = np.asarray(mesh_info["vertices"], dtype=np.float32)
    n_vertices = vertices.shape[0]
    logger.info("Loaded %d frames from %s (%d surface vertices)",
                total_frames, ds_cfg["raw_path"], n_vertices)

    # TrajectoryOptimizer is grid-based; the unstructured mesh is treated as
    # the degenerate grid (V, 1) over flattened vertex indices.
    fields = fields[:, :, None, None]                 # (N, V, 1, 1)

    domain_constraint = make_surface_constraint(n_vertices, str(device))

    # 2) Build the teacher reconstructor (external PhySense dependency).
    reconstructor = build_teacher_reconstructor(
        opt_cfg.get("teacher", "physense"),
        physense_repo=args.physense_repo,
        device=str(device),
        grid_shape=None,
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
            n_vertices, num_sensors, device, seed=args.seed + seg_idx)
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
            domain_constraint=domain_constraint)

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

        # SPF grid: the axis-aligned bounding box of the surface vertices at
        # unit resolution. Trajectory vertex indices are mapped back to
        # bounding-box-relative (x, y, z) coordinates; each segment's SPF has
        # shape (T, D1, D2, D3).
        bounds_min = vertices.min(axis=0)
        bounds_max = vertices.max(axis=0)
        spf_grid_shape = tuple(
            int(np.ceil(hi - lo)) + 1 for lo, hi in zip(bounds_min, bounds_max))
        traj_idx = trajectory.detach().cpu().numpy()[..., 0]
        traj_idx = np.clip(np.round(traj_idx), 0, n_vertices - 1).astype(int)
        traj_xyz = vertices[traj_idx] - bounds_min     # (T, k, 3), grid units
        traj_batch = traj_xyz[None]                    # (1, T, k, 3)
        spf = build_spf_sequence(traj_batch, spf_grid_shape, bandwidth)
        spf = np.asarray(spf, dtype=np.float32)        # (1, T, *grid)
        np.save(os.path.join(spf_dir, f"segment_{seg_idx:04d}.npy"), spf[0])
        spf_chunks.append(spf)
        final_loss = float(history[-1]["total"]) if history else float("nan")
        logger.info("Segment %d/%d done: trajectory -> %s (final loss %.6f)",
                    seg_idx + 1, len(starts), traj_path, final_loss)

    # Combined SPF tensor for sen-modality FTM training, shaped
    # (N_seg, T, D1, D2, D3) on the vertex bounding-box grid.
    if spf_chunks:
        combined = np.concatenate(spf_chunks, axis=0)
        combined_path = os.path.join(args.outdir, "spf_fields.npy")
        np.save(combined_path, combined)
        logger.info("Saved combined SPF fields %s -> %s",
                    combined.shape, combined_path)

    logger.info("All outputs written to %s", args.outdir)


if __name__ == "__main__":
    main()
