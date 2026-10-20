"""DyPSI: Dynamic Physics Sensing via Joint Field and Sensor-Trajectory Generation.

Official PyTorch implementation (NeurIPS 2026).
"""

from .models import (
    SineLayer,
    Continuous_Tucker_ssf,
    Tensor_inr_3D,
    Tensor_inr_4D,
    tucker_reconstruct_3d,
    tucker_reconstruct_4d,
    UNet,
    UNet_attention,
    Spatial_temporal_UNet,
    HistoryEncoder,
    FoldSpatialModeUNet,
)

__version__ = "1.0.0"

__all__ = [
    "SineLayer",
    "Continuous_Tucker_ssf",
    "Tensor_inr_3D",
    "Tensor_inr_4D",
    "tucker_reconstruct_3d",
    "tucker_reconstruct_4d",
    "UNet",
    "UNet_attention",
    "Spatial_temporal_UNet",
    "HistoryEncoder",
    "FoldSpatialModeUNet",
    "__version__",
]
