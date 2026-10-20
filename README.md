# DyPSI: Dynamic Physics Sensing via Joint Field and Sensor-Trajectory Generation

**NeurIPS 2026**

Official PyTorch implementation of *"DyPSI: Dynamic Physics Sensing via Joint Field and Sensor-Trajectory Generation"*.

DyPSI formulates dynamic physics sensing as **joint generative modeling of future physical fields and sensor trajectories conditioned on historical observations**:

$$p\big(P_{t+1:T},\, s_{t+1:T} \,\big|\, H_t\big)$$

Instead of committing to a fixed sensor placement, DyPSI *generates* where sensors should move next — together with the field they are meant to observe.

## Overview

DyPSI couples a shared continuous latent representation with a sequence-aware joint diffusion model. The method has three core ingredients:

1. **Sensor Position Fields (SPF)** — discrete sensor coordinates are lifted to continuous fields via anisotropic Gaussian kernels (Eq. 7), so that sensing locations become a modality that can be diffused jointly with physical fields; at inference time coordinates are read back with a top-k argmax.
2. **Shared Functional Tucker latent space** — physical fields and SPFs share the same set of spatial basis functions (a SIREN coordinate MLP). Only sample-/time-dependent core tensors enter the diffusion model, and the two modalities' cores are concatenated along the channel mode into a joint latent variable (Eq. 8–9).
3. **Joint diffusion with correlated noise** — an EDM-style diffusion over the joint latent, driven by Matérn-3/2 Gaussian-process temporally correlated noise (Eq. 10–11), makes the denoiser sequence-aware. Inference uses a 50-step second-order Heun sampler with soft anchoring on the observed prefix (Eq. 12, linearly annealed α).

Theoretical support: **Proposition 1** (the EDM risk linearly controls sequence-aggregated A-optimality) and **Proposition 2** (the GP prior controls trajectory total variation, guaranteeing kinematic feasibility).

Offline trajectory supervision is built with **Algorithm 2** (App. D): sequence-level optimization on top of any differentiable teacher reconstructor (PhySense by default), combining a reconstruction loss, a Matérn-3/2 GP temporal prior, and velocity/acceleration hinge constraints.

## Repository structure

```
github_dypsi/
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
├── configs/
│   ├── turbulent_flow.yaml          # config: 2D turbulent channel flow benchmark
│   ├── sea_temperature.yaml         # config: GLORYS12 sea surface temperature benchmark
│   └── car_aerodynamics.yaml        # config: OpenFOAM car surface pressure benchmark
├── dypsi/                           # core package
│   ├── models/
│   │   ├── ftm.py                   # Functional Tucker model (adapted from SDIFT)
│   │   ├── unet.py                  # spatio-temporal U-Net denoiser (adapted from SDIFT)
│   │   ├── wrappers.py              # FoldSpatialModeUNet for 3-spatial-mode latents (DyPSI core)
│   │   └── history_encoder.py       # history observation encoder (DyPSI core)
│   ├── diffusion/
│   │   ├── edm.py                   # EDM diffusion framework (adapted from SDIFT)
│   │   ├── gp_noise.py              # Matern-3/2 GP correlated noise (DyPSI core)
│   │   └── sampler.py               # conditional Heun sampler w/ soft anchoring (DyPSI core)
│   ├── data/
│   │   ├── spf.py                   # Sensor Position Fields: lift/read-back (DyPSI core)
│   │   ├── trajectory_optimizer.py  # Algorithm 2: offline trajectory supervision (DyPSI core)
│   │   ├── joint_dataset.py         # joint field + SPF segment dataset (DyPSI core)
│   │   └── datasets.py              # benchmark data loaders
│   ├── training/
│   │   ├── ftm_trainer.py           # Phase 2: FTM pre-training (both modalities)
│   │   └── diffusion_trainer.py     # Phase 3: joint diffusion training
│   ├── inference/
│   │   ├── joint_sampler.py         # Phase 4: conditional joint sampling
│   │   └── reconstruct.py           # external reconstructor evaluation
│   └── utils.py
└── scripts/
    ├── generate_data/               # Phase 1: benchmark data generation
    │   ├── generate_turbulent_flow.py
    │   ├── generate_sea_temperature.py
    │   ├── generate_car_aerodynamics.py
    │   └── teacher_adapters.py      # differentiable teacher reconstructor adapters (e.g. PhySense)
    ├── train/
    │   ├── train_ftm.py             # Phase 2 entry point
    │   └── train_joint_diffusion.py # Phase 3 entry point
    └── inference/
        ├── sample_conditional.py    # Phase 4: joint field + trajectory sampling
        └── reconstruct.py           # Phase 4: reconstruction & evaluation
```

Files marked *adapted from SDIFT* are modified from [SDIFT](https://github.com/OceanSTARLab/SDIFT); files marked *DyPSI core* are new contributions of this work.

## Installation

```bash
git clone https://github.com/<your-org>/github_dypsi.git
cd github_dypsi

conda create -n dypsi python=3.10 -y
conda activate dypsi

pip install -r requirements.txt
```

Requirements: **Python ≥ 3.10**, **PyTorch ≥ 2.0** (CUDA build recommended; training was done with bfloat16 on a single A100-80GB).

> **Optional dependency:** [PhySense](https://github.com/thuml/PhySense) is only needed if you want to use it as the differentiable teacher reconstructor for offline trajectory supervision (Algorithm 2) or as a baseline. DyPSI itself does not depend on it.

## Datasets

This repository does **not** ship data files. Please obtain the raw data from the original sources and place them under `data/` following the layout below (or adjust the paths in the YAML configs).

| Benchmark | Source | Raw data | Expected location |
|---|---|---|---|
| Turbulent flow | Senseiver dataset (Santos et al., 2023) | 2D turbulent channel flow, 128×48 | `data/turbulent_flow/` |
| Sea temperature | GLORYS12 reanalysis (Lellouche et al., 2021) | SST 360×180, train 1993–2019 / test 2020–2021, land mask applied | `data/sea_temperature/` |
| Car aerodynamics | OpenFOAM simulations on ShapeNet car meshes (Chang et al., 2015) | surface pressure on 95,428-vertex meshes | `data/car_aerodynamics/` |

## Usage

The full pipeline has four phases. Commands below are shown per benchmark; all phases are driven by the YAML configs in `configs/`.

### Phase 1 — Data generation

Generate training segments (fields + offline-optimized sensor trajectories via the teacher reconstructor):

```bash
python scripts/generate_data/generate_turbulent_flow.py   --config configs/turbulent_flow.yaml   --outdir data/turbulent_flow
python scripts/generate_data/generate_sea_temperature.py  --config configs/sea_temperature.yaml  --outdir data/sea_temperature
python scripts/generate_data/generate_car_aerodynamics.py --config configs/car_aerodynamics.yaml --outdir data/car_aerodynamics
```

### Phase 2 — FTM pre-training

Pre-train the Functional Tucker model for each modality (`phy` = physical field, `sen` = sensor position field); the two share spatial basis functions:

```bash
# turbulent flow
python scripts/train/train_ftm.py --config configs/turbulent_flow.yaml --modality phy --data_path data/turbulent_flow/fields.npy --outdir runs/tf_ftm_phy
python scripts/train/train_ftm.py --config configs/turbulent_flow.yaml --modality sen --data_path data/turbulent_flow/spf_fields.npy --outdir runs/tf_ftm_sen

# sea temperature
python scripts/train/train_ftm.py --config configs/sea_temperature.yaml --modality phy --data_path data/sea_temperature/fields.npy --outdir runs/sst_ftm_phy
python scripts/train/train_ftm.py --config configs/sea_temperature.yaml --modality sen --data_path data/sea_temperature/spf_fields.npy --outdir runs/sst_ftm_sen

# car aerodynamics
python scripts/train/train_ftm.py --config configs/car_aerodynamics.yaml --modality phy --data_path data/car_aerodynamics/fields.npy --outdir runs/car_ftm_phy
python scripts/train/train_ftm.py --config configs/car_aerodynamics.yaml --modality sen --data_path data/car_aerodynamics/spf_fields.npy --outdir runs/car_ftm_sen
```

Each run writes the FTM checkpoint (`runs/<run>/ftm_<modality>.pt`) and exports the per-sample core tensors (`runs/<run>/cores_<modality>.npy`) consumed by Phase 3.

### Phase 3 — Joint diffusion training

Train the joint diffusion model over the concatenated field/SPF core tensors (checkpoints from Phase 2 provide the shared bases and normalization statistics):

```bash
python scripts/train/train_joint_diffusion.py --config configs/turbulent_flow.yaml   --phy_core_path runs/tf_ftm_phy/cores_phy.npy  --sen_core_path runs/tf_ftm_sen/cores_sen.npy  --outdir runs/tf_diffusion
python scripts/train/train_joint_diffusion.py --config configs/sea_temperature.yaml  --phy_core_path runs/sst_ftm_phy/cores_phy.npy --sen_core_path runs/sst_ftm_sen/cores_sen.npy --outdir runs/sst_diffusion
python scripts/train/train_joint_diffusion.py --config configs/car_aerodynamics.yaml --phy_core_path runs/car_ftm_phy/cores_phy.npy --sen_core_path runs/car_ftm_sen/cores_sen.npy --outdir runs/car_diffusion
```

Default training hyper-parameters (App. C.3): Adam, lr 2e-4, batch 16 segments, 200k iterations, EDM noise ln σ ~ N(−1.2, 1.2²), σ_data = 0.5, segment length T_seg = 16, K_obs = 4 observed frames, bfloat16. Training takes roughly 14 / 22 / 28 hours (turbulent flow / SST / car) on a single A100-80GB.

### Phase 4 — Inference

Joint conditional sampling of future fields and sensor trajectories from observed history (50-step Heun sampler with soft prefix anchoring), followed by reconstruction from the generated sensor observations:

```bash
# turbulent flow
python scripts/inference/sample_conditional.py --config configs/turbulent_flow.yaml   --denoiser_ckpt runs/tf_diffusion/checkpoints/ema_final.pth  --norm_stats runs/tf_diffusion/norm_stats.npz  --ftm_ckpt_phy runs/tf_ftm_phy/ftm_phy.pt  --ftm_ckpt_sen runs/tf_ftm_sen/ftm_sen.pt  --observed_cores data/turbulent_flow/observed_cores.npy  --outdir results/turbulent_flow
python scripts/inference/reconstruct.py        --config configs/turbulent_flow.yaml   --reconstructor physense --physense_repo /path/to/PhySense --field_path data/turbulent_flow/test_fields.npy --placements results/turbulent_flow/placements.npz --outdir results/turbulent_flow/recon

# sea temperature
python scripts/inference/sample_conditional.py --config configs/sea_temperature.yaml  --denoiser_ckpt runs/sst_diffusion/checkpoints/ema_final.pth --norm_stats runs/sst_diffusion/norm_stats.npz --ftm_ckpt_phy runs/sst_ftm_phy/ftm_phy.pt --ftm_ckpt_sen runs/sst_ftm_sen/ftm_sen.pt --observed_cores data/sea_temperature/observed_cores.npy --outdir results/sea_temperature
python scripts/inference/reconstruct.py        --config configs/sea_temperature.yaml  --reconstructor physense --physense_repo /path/to/PhySense --field_path data/sea_temperature/test_fields.npy --placements results/sea_temperature/placements.npz --outdir results/sea_temperature/recon

# car aerodynamics
python scripts/inference/sample_conditional.py --config configs/car_aerodynamics.yaml --denoiser_ckpt runs/car_diffusion/checkpoints/ema_final.pth --norm_stats runs/car_diffusion/norm_stats.npz --ftm_ckpt_phy runs/car_ftm_phy/ftm_phy.pt --ftm_ckpt_sen runs/car_ftm_sen/ftm_sen.pt --observed_cores data/car_aerodynamics/observed_cores.npy --outdir results/car_aerodynamics
python scripts/inference/reconstruct.py        --config configs/car_aerodynamics.yaml --reconstructor physense --physense_repo /path/to/PhySense --field_path data/car_aerodynamics/test_fields.npy --placements results/car_aerodynamics/placements.npz --outdir results/car_aerodynamics/recon
```

## Results

Relative L2 error of reconstruction from DyPSI-generated trajectories vs. PhySense with optimized sensor placement (PhySense-opt), Tables 1–2 of the paper:

| Benchmark | # Sensors | DyPSI | PhySense-opt | Improvement |
|---|---:|---:|---:|---:|
| Turbulent flow | 30 | 0.1826 | 0.2157 | 15.35% |
| Sea temperature (SST) | 15 | 0.0402 | 0.0439 | 8.43% |
| Car aerodynamics | 15 | 0.0311 | 0.0386 | 19.43% |

On the car benchmark DyPSI outperforms PhySense-opt by **18.4% on average** across sensor budgets.

## Method details

### Sensor Position Fields (SPF)
Each sensor location is encoded as an anisotropic Gaussian kernel on the domain (Eq. 7), turning a set of discrete coordinates into a continuous field that shares the same spatial discretization — and hence the same latent machinery — as the physical field. After sampling, sensor coordinates are recovered via top-k argmax on the generated SPF.

### Shared Functional Tucker latent space
A SIREN coordinate MLP provides spatial basis functions shared by the physical field and the SPF. A field snapshot is compressed to a small core tensor by projecting onto these bases (Functional Tucker decomposition); the two modalities' cores are concatenated along the channel mode to form the joint latent variable (Eq. 8–9). The diffusion model operates entirely on these compact cores, never in pixel space.

### Matérn-3/2 GP correlated noise
Instead of i.i.d. Gaussian noise, the diffusion process adds noise drawn from a Gaussian process with a Matérn-3/2 temporal kernel (Eq. 10–11). This couples the noise levels across time steps, making the denoiser sequence-aware and — per Proposition 2 — controlling the total variation of generated trajectories so they remain kinematically feasible.

### Soft-anchored conditional sampling
Inference uses a 50-step second-order Heun sampler. The observed prefix (K_obs frames encoded into cores) is softly anchored at every step: generated prefix cores are pulled toward the observations with a weight α that is linearly annealed to zero over sampling (Eq. 12), giving strict consistency early and generative freedom late.

### 3D unstructured meshes (car aerodynamics)
The car benchmark lives on a 95,428-vertex surface mesh. The surface-mesh embedding is precomputed via the signed-distance projection of the original ShapeNet vertices onto a regular 3D grid (paper App. C.2), so both modalities factorize into **4-mode Functional Tucker cores** `(R1, R2, R3, R4)` (phy `[24, 24, 24, 16]`, sen `[8, 24, 24, 16]` in `configs/car_aerodynamics.yaml`) handled by `Tensor_inr_4D` / `FTMTrainer` with 4-entry ranks.

*Release implementation note:* the joint denoiser remains the 2D `Spatial_temporal_UNet`. The third spatial rank mode is folded into the channel axis by `dypsi.models.wrappers.FoldSpatialModeUNet` — `(B, T, C, R2, R3, R4) → (B, T, C·R4, R2, R3)` and unfolded after denoising. With the car ranks this gives a square 24×24 latent grid (`img_resolution = R2 = 24`, compatible with `channel_mult: [1, 2, 4, 4]` → 24→12→6→3, `attn_resolutions: [12, 6]`) and `in_channels = (24+8)·16 = 512` (computed automatically by the training/sampling scripts). Enable it with `diffusion.unet.fold_spatial_modes: true` in the config; decoding at inference uses the `Tensor_inr_4D` sampling mode over a 3D `grid_shape`.

## Acknowledgments

- Code adapted from [SDIFT](https://github.com/OceanSTARLab/SDIFT) (Chen et al., 2025, arXiv:2505.09284) — FTM, U-Net and EDM components.
- [PhySense](https://github.com/thuml/PhySense) (Ma et al., 2025, arXiv:2505.18190) — default teacher reconstructor and baseline.
- Turbulent flow data from the [Senseiver](https://github.com/google/senseiver) dataset (Santos et al., 2023).
- EDM framework from Karras et al., 2022.
- Sea surface temperature from the GLORYS12 reanalysis (Lellouche et al., 2021); car geometries from ShapeNet (Chang et al., 2015); aerodynamic simulations with OpenFOAM.

## Citation

```bibtex
@article{dypsi2026,
  title   = {DyPSI: Dynamic Physics Sensing via Joint Field and Sensor-Trajectory Generation},
  author  = {Anonymous},
  journal = {Advances in Neural Information Processing Systems (NeurIPS)},
  year    = {2026}
}
```

## License

This project is released under the MIT License — see [LICENSE](LICENSE).
