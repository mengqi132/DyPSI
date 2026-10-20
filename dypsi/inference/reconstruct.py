# Core component of DyPSI (NeurIPS 2026).
# Requires an external reconstructor (e.g. PhySense: https://github.com/thuml/PhySense) installed separately.
"""Evaluation wrapper around an external differentiable reconstructor.

The reconstructor must implement
``reconstruct(sensor_values, sensor_coords, grid_shape, num_steps)`` ->
dense field (1, H, W, C), following the ``ReconstructorAdapter`` protocol in
``dypsi.data.trajectory_optimizer``.
"""

import numpy as np
import torch

from dypsi.utils import bilinear_sample, get_device, relative_l2

__all__ = ["evaluate_placement", "compare_placements"]

_PROTOCOL_HINT = (
    "The provided reconstructor does not implement the expected interface: "
    "`reconstruct(sensor_values, sensor_coords, grid_shape, num_steps)` "
    "returning a dense field of shape (1, H, W, C). Wrap your external model "
    "(e.g. PhySense) in an adapter following "
    "dypsi.data.trajectory_optimizer.ReconstructorAdapter; see the scripts/ "
    "directory for an example."
)


def _check_reconstructor(reconstructor) -> None:
    """Validate that ``reconstructor`` follows the ReconstructorAdapter protocol."""
    fn = getattr(reconstructor, "reconstruct", None)
    if not callable(fn):
        raise NotImplementedError(_PROTOCOL_HINT)


def _as_bhwc(field: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Normalize a field to shape (1, H, W, C)."""
    field = torch.as_tensor(field, dtype=torch.float32, device=device)
    if field.dim() == 2:        # (H, W) -> (1, H, W, 1)
        field = field[None, ..., None]
    elif field.dim() == 3:      # (H, W, C) -> (1, H, W, C)
        field = field[None]
    if field.dim() != 4:
        raise ValueError(f"Expected field of shape (H,W), (H,W,C) or (1,H,W,C); got {tuple(field.shape)}")
    return field


def evaluate_placement(reconstructor, field: torch.Tensor, sensor_coords: np.ndarray,
                       num_euler_steps: int = 10, device: str = "auto") -> dict:
    """Reconstruct a dense field from sparse readings at ``sensor_coords``.

    Takes sparse observations of the ground-truth field via bilinear sampling,
    runs the external reconstructor, and computes error metrics.

    Args:
        reconstructor: Object implementing the ``ReconstructorAdapter``
            protocol (see module docstring).
        field: Ground-truth field, (H, W), (H, W, C) or (1, H, W, C).
        sensor_coords: Sensor pixel coordinates, (k, 2) in (x, y) grid units.
        num_euler_steps: Number of Euler steps for the reconstructor's sampler.
        device: Device preference, see ``dypsi.utils.get_device``.

    Returns:
        Dict with ``'reconstruction'`` (H, W) NumPy array (channel 0),
        ``'mse'``, ``'relative_l2'``, ``'r2'``, ``'max_error'`` and
        ``'sensor_coords'``.

    Raises:
        NotImplementedError: If ``reconstructor`` does not satisfy the
            ``ReconstructorAdapter`` protocol.
    """
    _check_reconstructor(reconstructor)
    dev = get_device(device)

    gt = _as_bhwc(field, dev)  # (1, H, W, C)
    _, H, W, C = gt.shape
    coords_np = np.asarray(sensor_coords, dtype=np.float32).reshape(-1, 2)
    coords_t = torch.as_tensor(coords_np, dtype=torch.float32, device=dev)

    with torch.no_grad():
        # Sparse observations from the ground-truth field: (1, k, C).
        obs = bilinear_sample(gt, coords_t.unsqueeze(0))

        recon = reconstructor.reconstruct(
            sensor_values=obs,
            sensor_coords=coords_t,
            grid_shape=(H, W),
            num_steps=num_euler_steps,
        )
        recon = _as_bhwc(recon, dev)
        if recon.shape[1:3] != (H, W):
            raise ValueError(
                f"Reconstructor returned shape {tuple(recon.shape)}; expected grid {(H, W)}. "
                + _PROTOCOL_HINT
            )

        mse = (recon - gt).square().mean().item()
        rel_l2 = relative_l2(recon, gt).item()

        gt_flat = gt.reshape(-1)
        rec_flat = recon.reshape(-1)
        ss_res = ((rec_flat - gt_flat) ** 2).sum().item()
        ss_tot = ((gt_flat - gt_flat.mean()) ** 2).sum().item()
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        max_error = torch.abs(recon - gt).max().item()

    return {
        "reconstruction": recon[0, :, :, 0].cpu().numpy(),  # (H, W), channel 0
        "mse": mse,
        "relative_l2": rel_l2,
        "r2": r2,
        "max_error": max_error,
        "sensor_coords": coords_np,
    }


def compare_placements(reconstructor, field: torch.Tensor,
                       placements: "dict[str, np.ndarray]", **kw) -> dict:
    """Evaluate several sensor placements on the same field and print a table.

    Args:
        reconstructor: Object implementing the ``ReconstructorAdapter``
            protocol.
        field: Ground-truth field, (H, W), (H, W, C) or (1, H, W, C).
        placements: Mapping of placement name -> sensor coordinates (k, 2),
            e.g. ``{'dypsi': coords, 'static': coords, 'random': coords}``.
        **kw: Forwarded to :func:`evaluate_placement` (e.g. ``num_euler_steps``,
            ``device``).

    Returns:
        Dict mapping each placement name to its :func:`evaluate_placement`
        result dict.
    """
    results = {}
    for name, coords in placements.items():
        print(f"[compare_placements] evaluating '{name}' ({len(coords)} sensors)...")
        results[name] = evaluate_placement(reconstructor, field, coords, **kw)

    header = f"{'Placement':<20} {'MSE':>12} {'R2':>10} {'Relative L2':>13} {'Max Error':>11}"
    print("\n" + "=" * len(header))
    print("Reconstruction comparison")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for name, res in results.items():
        print(f"{name:<20} {res['mse']:>12.6f} {res['r2']:>10.4f} "
              f"{res['relative_l2']:>13.6f} {res['max_error']:>11.6f}")
    print("=" * len(header))

    best_mse = min(results, key=lambda n: results[n]["mse"])
    best_r2 = max(results, key=lambda n: results[n]["r2"])
    print(f"Best MSE: {best_mse} ({results[best_mse]['mse']:.6f}) | "
          f"Best R2: {best_r2} ({results[best_r2]['r2']:.6f})")
    return results
