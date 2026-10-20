# Core component of DyPSI (NeurIPS 2026).
"""Teacher reconstructor adapters for DyPSI data generation.

Wraps PhySense behind the ``dypsi.data.ReconstructorAdapter`` protocol.
PhySense is an external dependency, imported lazily inside methods so this
module can be imported without it installed.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

PHYSENSE_INSTALL_HINT = (
    "PhySense is an external dependency: https://github.com/thuml/PhySense "
    "(Ma et al., 2025). Install it separately and pass --physense_repo <path>."
)


class PhySenseAdapter:
    """Expose a PhySense model through the ReconstructorAdapter protocol.

    Used by ``dypsi.data.TrajectoryOptimizer`` and
    ``dypsi.inference.reconstruct.evaluate_placement``.

    Parameters
    ----------
    physense_repo : str or None
        Path to a local PhySense clone, prepended to ``sys.path``.
    ckpt_path : str or None
        Optional pretrained PhySense checkpoint.
    grid_shape : sequence of int or None
        Spatial resolution of the reconstructed fields.
    n_channels : int
        Number of physical field channels.
    device : str
        Torch device string (e.g. "cuda", "cpu").
    """

    def __init__(self, physense_repo=None, ckpt_path=None, grid_shape=None,
                 n_channels=1, device="cpu"):
        self.physense_repo = physense_repo
        self.ckpt_path = ckpt_path
        self.grid_shape = tuple(grid_shape) if grid_shape is not None else None
        self.n_channels = n_channels
        self.device = device
        self._model = None
        self._torch = None

    # Lazy external imports
    def _import_physense(self):
        """Import torch and PhySense, adding ``physense_repo`` to ``sys.path``."""
        if self.physense_repo is not None:
            repo = os.path.abspath(self.physense_repo)
            if repo not in sys.path:
                sys.path.insert(0, repo)
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "PyTorch is required for the PhySense teacher reconstructor."
            ) from exc
        try:
            import physense  # noqa: F401
        except ImportError as exc:
            raise ImportError(PHYSENSE_INSTALL_HINT) from exc
        self._torch = torch
        return torch, physense

    def _build_model(self):
        """Instantiate the PhySense model and optionally load a checkpoint."""
        if self._model is not None:
            return self._model
        torch, physense = self._import_physense()
        # PhySense exposes its model zoo differently across releases; resolve
        # the first available constructor rather than pinning one API.
        builder = None
        for attr in ("PhySense", "build_model", "PhySenseModel"):
            builder = getattr(physense, attr, None)
            if builder is not None:
                break
        if builder is None:
            raise ImportError(
                "Could not locate a PhySense model constructor. "
                + PHYSENSE_INSTALL_HINT
            )
        model = builder(grid_shape=self.grid_shape, n_channels=self.n_channels)
        if self.ckpt_path is not None:
            state = torch.load(self.ckpt_path, map_location="cpu")
            model.load_state_dict(state.get("state_dict", state))
        model.to(self.device)
        model.eval()
        self._model = model
        return model

    # ReconstructorAdapter protocol
    @property
    def positional_encoding(self):
        """Coordinate embedding used by PhySense to query sensor positions."""
        model = self._build_model()
        encoding = getattr(model, "positional_encoding", None)
        if encoding is None:
            raise AttributeError(
                "The loaded PhySense model does not expose "
                "'positional_encoding'. " + PHYSENSE_INSTALL_HINT
            )
        return encoding

    def init_state(self, initial_frame):
        """Initialize the latent state from the first observed frame."""
        torch, _ = self._import_physense()
        model = self._build_model()
        with torch.no_grad():
            frame = torch.as_tensor(initial_frame, dtype=torch.float32,
                                    device=self.device)
            return model.encode(frame)

    def step(self, z, t, coords, sensor_values, sensor_pos):
        """Advance the latent state one step given sparse sensor observations.

        z: latent state (B, H, W, C). coords: sensor coordinates (B, k, 2)
        in (x, y) grid units, differentiable. Returns the vector-field
        increment dz/dt, same shape as z.
        """
        model = self._build_model()
        return model.step(z, t, coords, sensor_values, sensor_pos)

    def reconstruct(self, sensor_values, sensor_coords, grid_shape,
                    num_steps=10):
        """Dense reconstruction from sparse readings (ReconstructorAdapter).

        Integrates :meth:`step` with ``num_steps`` Euler steps from Gaussian
        noise. sensor_values: (B, k, C); sensor_coords: (B, k, 2) in (x, y)
        grid units. Returns the dense field (B, H, W, C).
        """
        torch, _ = self._import_physense()
        self._build_model()
        from dypsi.utils import bilinear_sample

        with torch.no_grad():
            values = torch.as_tensor(sensor_values, dtype=torch.float32,
                                     device=self.device)
            coords = torch.as_tensor(sensor_coords, dtype=torch.float32,
                                     device=self.device)
            if values.dim() == 2:
                values = values.unsqueeze(0)          # (1, k, C)
            if coords.dim() == 2:
                coords = coords.unsqueeze(0)          # (1, k, 2)

            H, W = (int(s) for s in grid_shape)
            pos = self.positional_encoding
            pos = torch.as_tensor(pos, dtype=torch.float32, device=self.device)
            if pos.dim() == 3:
                pos = pos.unsqueeze(0)                # (1, H, W, P)
            sensor_pos = bilinear_sample(pos, coords)  # (B, k, P)

            B = values.shape[0]
            z = torch.randn(B, H, W, self.n_channels, device=self.device)
            dt = 1.0 / int(num_steps)
            for i in range(int(num_steps)):
                t = torch.full((B,), i / int(num_steps), device=self.device)
                z = z + self.step(z, t, coords, values, sensor_pos) * dt
        return z


def build_teacher_reconstructor(name, physense_repo=None, device="cpu", **kwargs):
    """Factory for teacher reconstructors used in trajectory optimization.

    name: teacher identifier from the config; currently only "physense".
    Extra kwargs are forwarded to the adapter constructor.
    """
    name = (name or "").lower()
    if name == "physense":
        return PhySenseAdapter(physense_repo=physense_repo, device=device, **kwargs)
    raise ValueError(
        f"Unknown teacher reconstructor '{name}'. Supported: ['physense']."
    )
