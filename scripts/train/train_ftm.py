# Core component of DyPSI (NeurIPS 2026).
# Adapted from SDIFT (Chen et al., 2025): https://github.com/OceanSTARLab/SDIFT
"""Train the shared-basis Functional Tensor-Module (FTM) for one modality
('phy' physical fields or 'sen' sensor placement fields) of a dataset.
"""

import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np

from dypsi.training.ftm_trainer import FTMTrainConfig, FTMTrainer
from dypsi.utils import get_device, load_config, set_seed

logger = logging.getLogger("dypsi.train_ftm")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the shared-basis FTM for the 'phy' or 'sen' "
                    "modality.")
    parser.add_argument("--config", required=True,
                        help="Path to the dataset YAML config.")
    parser.add_argument("--modality", required=True, choices=["phy", "sen"],
                        help="Which modality to factorize: 'phy' (physical "
                             "fields) or 'sen' (sensor placement fields).")
    parser.add_argument("--data_path", required=True,
                        help="Field data as .npy or .pth, shaped (N, T, *spatial) "
                             "or (T, *spatial) with 2 or 3 spatial axes (the "
                             "latter for 3D benchmarks, e.g. car aerodynamics).")
    parser.add_argument("--outdir", required=True,
                        help="Directory for checkpoints and training history.")
    parser.add_argument("--device", default="auto",
                        help="Device preference: 'auto', 'cpu', 'cuda', ...")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed.")
    return parser.parse_args()


def load_fields(data_path):
    """Load field data from .npy or .pth and normalize to (N, T, H, W)."""
    ext = os.path.splitext(data_path)[1].lower()
    if ext == ".npy":
        data = np.load(data_path)
    elif ext == ".pth":
        import torch

        data = torch.load(data_path, map_location="cpu")
        if isinstance(data, dict):
            for key in ("fields", "data", "tensor"):
                if key in data:
                    data = data[key]
                    break
            else:
                raise ValueError(
                    f"Could not find a field tensor in {data_path}; "
                    f"available keys: {sorted(data.keys())}")
        data = np.asarray(data)
    else:
        raise ValueError(
            f"Unsupported data extension '{ext}'; expected .npy or .pth.")
    data = np.asarray(data, dtype=np.float32)
    # (T, H, W) -> single 2D-spatial sequence. A 3D-spatial sequence
    # (T, H, W, Z) is ambiguous with (N, T, H, W), so it must carry an
    # explicit leading batch axis: (1, T, H, W, Z).
    if data.ndim == 3:
        data = data[None]
    if data.ndim not in (4, 5):
        raise ValueError(
            f"Expected data of shape (N, T, H, W), (T, H, W) or "
            f"(N, T, H, W, Z), got {data.shape}.")
    return data


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    cfg = load_config(args.config)
    set_seed(args.seed)
    device = get_device(args.device)

    ftm_cfg = cfg["ftm"][args.modality]
    logger.info("Training FTM modality='%s' ranks=%s omega=%s (device=%s)",
                args.modality, ftm_cfg["ranks"], ftm_cfg["omega"], device)

    data = load_fields(args.data_path)
    logger.info("Loaded data %s from %s", data.shape, args.data_path)

    train_cfg = FTMTrainConfig(
        ranks=list(ftm_cfg["ranks"]),
        omega=ftm_cfg["omega"],
        lr=ftm_cfg["lr"],
        weight_decay=ftm_cfg["weight_decay"],
        tv_weight=ftm_cfg["tv_weight"],
        max_iters=ftm_cfg["max_iters"],
        eval_interval=ftm_cfg["eval_interval"],
        grad_clip=ftm_cfg["grad_clip"],
        device=str(device),
    )

    os.makedirs(args.outdir, exist_ok=True)
    trainer = FTMTrainer(data, train_cfg)
    ckpt_name = f"ftm_{args.modality}.pt"
    history = trainer.train(args.outdir, ckpt_name)

    # Export per-sample cores as .npy; inputs of the joint diffusion
    # training (--phy_core_path / --sen_core_path).
    cores_path = os.path.join(args.outdir, f"cores_{args.modality}.npy")
    np.save(cores_path,
            trainer.core.detach().cpu().numpy().astype(np.float32))

    history_path = os.path.join(
        args.outdir, f"ftm_{args.modality}_history.json")
    with open(history_path, "w", encoding="utf-8") as fh:
        json.dump(history, fh, indent=2, default=float)
    logger.info("Checkpoint -> %s ; cores -> %s ; history -> %s",
                os.path.join(args.outdir, ckpt_name), cores_path,
                history_path)


if __name__ == "__main__":
    main()
