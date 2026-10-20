# Core component of DyPSI (NeurIPS 2026).
"""Reconstruction benchmark comparing sensor placements on a ground-truth field.

Compares DyPSI-optimized placements against static and random baselines by
rolling out an external reconstructor from the sparse readings of each
placement. The reconstructor (PhySense, https://github.com/thuml/PhySense)
is an external dependency, NOT part of DyPSI; it is imported lazily and used
only as an evaluation oracle.
"""

import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np

from dypsi.inference.reconstruct import compare_placements
from dypsi.utils import get_device, load_config, set_seed
from scripts.generate_data.teacher_adapters import build_teacher_reconstructor

logger = logging.getLogger("dypsi.reconstruct")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare sensor placements (dypsi/static/random) via an "
                    "external reconstructor.")
    parser.add_argument("--config", required=True,
                        help="Path to the dataset YAML config.")
    parser.add_argument("--reconstructor", default="physense",
                        choices=["physense"],
                        help="External reconstructor used as the evaluation "
                             "oracle. Currently only 'physense'.")
    parser.add_argument("--physense_repo", default=None,
                        help="Path to a local clone of the PhySense repository.")
    parser.add_argument("--field_path", required=True,
                        help="Ground-truth field frame(s) as .npy or .pth.")
    parser.add_argument("--placements", required=True,
                        help=".npz or .pth containing named coordinate sets, "
                             "e.g. keys 'dypsi', 'static', 'random'.")
    parser.add_argument("--outdir", required=True,
                        help="Directory for the comparison results (.json).")
    parser.add_argument("--num_euler_steps", type=int, default=10,
                        help="Reconstructor Euler integration steps.")
    parser.add_argument("--device", default="auto",
                        help="Device preference: 'auto', 'cpu', 'cuda', ...")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed.")
    return parser.parse_args()


def load_field(field_path):
    """Load the ground-truth field from .npy or .pth."""
    ext = os.path.splitext(field_path)[1].lower()
    if ext == ".npy":
        return np.load(field_path)
    if ext == ".pth":
        import torch

        data = torch.load(field_path, map_location="cpu")
        if isinstance(data, dict):
            for key in ("field", "fields", "data"):
                if key in data:
                    data = data[key]
                    break
            else:
                raise ValueError(
                    f"Could not find a field tensor in {field_path}; "
                    f"available keys: {sorted(data.keys())}")
        return np.asarray(data)
    raise ValueError(
        f"Unsupported field extension '{ext}'; expected .npy or .pth.")


def load_placements(placements_path):
    """Load named sensor-coordinate sets from .npz or .pth."""
    ext = os.path.splitext(placements_path)[1].lower()
    if ext == ".npz":
        with np.load(placements_path) as archive:
            placements = {name: archive[name] for name in archive.files}
    elif ext == ".pth":
        import torch

        data = torch.load(placements_path, map_location="cpu")
        if not isinstance(data, dict):
            raise ValueError(
                f"Expected a dict of placements in {placements_path}.")
        placements = {name: np.asarray(coords) for name, coords in data.items()}
    else:
        raise ValueError(
            f"Unsupported placements extension '{ext}'; expected .npz or .pth.")
    if not placements:
        raise ValueError(f"No placements found in {placements_path}.")
    return placements


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    cfg = load_config(args.config)
    set_seed(args.seed)
    device = get_device(args.device)
    os.makedirs(args.outdir, exist_ok=True)

    field = load_field(args.field_path)
    placements = load_placements(args.placements)
    logger.info("Ground-truth field %s; placements: %s",
                field.shape,
                {k: tuple(np.asarray(v).shape) for k, v in placements.items()})

    # PhySense is external (not part of DyPSI); used only as the evaluation
    # oracle. Import failure surfaces the install hint.
    reconstructor = build_teacher_reconstructor(
        args.reconstructor,
        physense_repo=args.physense_repo,
        device=str(device),
        grid_shape=cfg["dataset"].get("grid_shape"),
        n_channels=cfg["dataset"]["n_channels"],
    )

    results = compare_placements(
        reconstructor, field, placements,
        num_euler_steps=args.num_euler_steps, device=str(device))

    print("=== Reconstruction comparison "
          f"({args.reconstructor}, euler_steps={args.num_euler_steps}) ===")
    metric_names = sorted({
        metric for metrics in results.values() for metric in metrics})
    header = f"{'placement':<16}" + "".join(f"{m:>16}" for m in metric_names)
    print(header)
    print("-" * len(header))
    for name, metrics in results.items():
        row = f"{name:<16}" + "".join(
            f"{metrics.get(m, float('nan')):>16.6f}" for m in metric_names)
        print(row)

    out_path = os.path.join(args.outdir, "placement_comparison.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, default=float)
    logger.info("Comparison results -> %s", out_path)


if __name__ == "__main__":
    main()
