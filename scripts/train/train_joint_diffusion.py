# Core component of DyPSI (NeurIPS 2026).
"""Train the DyPSI joint field-and-sensor-trajectory diffusion prior.

Trains the joint diffusion model over concatenated phy/sen FTM core
sequences with EDM preconditioning (Karras et al., 2022) and a Matern-3/2
GP temporal prior.
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from dypsi.training.diffusion_trainer import train_joint_diffusion
from dypsi.utils import get_device, load_config, set_seed

logger = logging.getLogger("dypsi.train_joint_diffusion")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the DyPSI joint diffusion prior over phy/sen FTM "
                    "core sequences.")
    parser.add_argument("--config", required=True,
                        help="Path to the dataset YAML config.")
    parser.add_argument("--phy_core_path", required=True,
                        help="Path to the physical-field FTM core sequences "
                             "(.npy), produced from the phy-modality FTM.")
    parser.add_argument("--sen_core_path", required=True,
                        help="Path to the sensor-placement FTM core sequences "
                             "(.npy), produced from the sen-modality FTM.")
    parser.add_argument("--outdir", required=True,
                        help="Directory for denoiser checkpoints and logs.")
    parser.add_argument("--device", default="auto",
                        help="Device preference: 'auto', 'cpu', 'cuda', ...")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed.")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    cfg = load_config(args.config)
    set_seed(args.seed)
    device = get_device(args.device)
    os.makedirs(args.outdir, exist_ok=True)

    # Config dict layout expected by
    # dypsi.training.diffusion_trainer.train_joint_diffusion:
    # {seed, outdir, device, data, unet, edm, gp, history, train}.
    joint_cfg = cfg["joint"]
    diff_cfg = cfg.get("diffusion", {})
    ranks_phy = list(cfg["ftm"]["phy"]["ranks"])
    ranks_sen = list(cfg["ftm"]["sen"]["ranks"])
    if ranks_phy[1:] != ranks_sen[1:]:
        raise ValueError(
            f"Spatial FTM ranks must match for joint stacking: "
            f"phy {ranks_phy} vs sen {ranks_sen}")

    unet_cfg = dict(diff_cfg.get("unet", {}))
    # Latent grid is (R2, R3); img_resolution only steers the per-level
    # resolution bookkeeping (attention resolutions), so use the larger rank.
    unet_cfg.setdefault("img_resolution", max(ranks_phy[1], ranks_phy[2]))
    in_channels = ranks_phy[0] + ranks_sen[0]
    if len(ranks_phy) == 4:
        # 3-spatial-mode cores (car aerodynamics) fold R4 into the channel axis.
        unet_cfg["fold_spatial_modes"] = True
        unet_cfg["img_resolution"] = ranks_phy[1]  # R2 (= R3 for the car grid)
        in_channels *= ranks_phy[3]
    unet_cfg["in_channels"] = in_channels

    train_cfg = {
        "seed": args.seed,
        "outdir": args.outdir,
        "device": str(device),
        "data": {
            "phy_core_path": args.phy_core_path,
            "sen_core_path": args.sen_core_path,
            "seg_len": int(joint_cfg.get("seg_len", 16)),
            "stride": int(joint_cfg.get("stride", 5)),
            "normalize": True,
        },
        "unet": unet_cfg,
        "edm": diff_cfg.get("edm", {}),
        "gp": diff_cfg.get("gp", {}),
        "history": diff_cfg.get("history", {"enabled": False}),
        "train": diff_cfg.get("training", {}),
    }
    logger.info("Starting joint diffusion training (dataset=%s, outdir=%s)",
                cfg["dataset"]["name"], args.outdir)

    ckpt_path = train_joint_diffusion(train_cfg)
    logger.info("Joint diffusion training finished. Final denoiser "
                "checkpoint -> %s", ckpt_path)


if __name__ == "__main__":
    main()
