# History encoder for conditional joint generation.
# Core component of DyPSI (NeurIPS 2026).

import math
from typing import Optional

import torch
from torch import nn


def sinusoidal_time_encoding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal encoding of scalar timestamps: (B, L) -> (B, L, dim).

    Args:
        dim: Output encoding dimension. Must be even.
    """
    if dim % 2 != 0:
        raise ValueError("sinusoidal_time_encoding requires an even dim.")
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float32,
                                                        device=t.device) / half)
    angles = t.unsqueeze(-1) * freqs  # (B, L, half)
    return torch.cat([angles.sin(), angles.cos()], dim=-1)


class HistoryEncoder(nn.Module):
    """Encodes the observed prefix {(s_tau, y_tau)} into a conditioning
    embedding h_hist.

    Each observation of the prefix is embedded by a shared MLP over the
    concatenation of sensor coordinates, observed values, and a sinusoidal
    time encoding. The embedded sequence is processed by a Transformer
    encoder and mean-pooled into a single vector of shape (B, hidden_dim),
    passed to the joint denoiser through its ``label_dim`` conditioning
    channel.

    Args:
        coord_dim: Dimension of each sensor coordinate s_tau (2 for 2D fields).
        value_dim: Dimension of each single-sensor observation y_tau (1 for a
            scalar field).
        sensor_dim: Number of sensors k observed per time step; each time step
            contributes k * value_dim scalar values.
        hidden_dim: Transformer width and output embedding dimension.
        num_layers: Number of Transformer encoder layers.
        num_heads: Number of attention heads per layer.
        max_len: Maximum supported prefix length L (defines the learnable
            positional embedding table).
        time_dim: Dimension of the sinusoidal time encoding. Must be even.
        dropout: Dropout used inside the Transformer encoder layers.
    """

    def __init__(self,
                 coord_dim: int = 2,
                 value_dim: int = 1,
                 sensor_dim: int = 1,
                 hidden_dim: int = 256,
                 num_layers: int = 4,
                 num_heads: int = 4,
                 max_len: int = 64,
                 time_dim: int = 32,
                 dropout: float = 0.0) -> None:
        super().__init__()
        if time_dim % 2 != 0:
            raise ValueError("time_dim must be even.")

        self.coord_dim = coord_dim
        self.value_dim = value_dim
        self.sensor_dim = sensor_dim
        self.hidden_dim = hidden_dim
        self.max_len = max_len
        self.time_dim = time_dim

        obs_dim = coord_dim + sensor_dim * value_dim

        # Shared MLP embedding each observation (coords + values + time enc.).
        self.input_proj = nn.Sequential(
            nn.Linear(obs_dim + time_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Learnable positional embeddings over the prefix index.
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, hidden_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=4 * hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.final_norm = nn.LayerNorm(hidden_dim)

    def forward(self,
                coords: torch.Tensor,
                values: torch.Tensor,
                t: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode the observed prefix into h_hist.

        Args:
            coords: Sensor coordinates, shape (B, L, coord_dim).
            values: Observed values, shape (B, L, sensor_dim * value_dim) or
                (B, L, k, C), which is flattened internally to (B, L, k * C).
            t: Observation timestamps, shape (B, L) or (B, L, 1).
            mask: Optional boolean padding mask of shape (B, L); True marks
                padded positions that are ignored by attention and pooling.

        Returns:
            Conditioning embedding h_hist of shape (B, hidden_dim).
        """
        if values.dim() == 4:
            values = values.flatten(2)  # (B, L, k * C)

        if t.dim() == 3:
            t = t.squeeze(-1)  # (B, L)

        B, L, _ = coords.shape
        if L > self.max_len:
            raise ValueError(f"Prefix length {L} exceeds max_len={self.max_len}.")

        time_enc = sinusoidal_time_encoding(t, self.time_dim)  # (B, L, time_dim)
        x = torch.cat([coords, values, time_enc], dim=-1)  # (B, L, obs_dim + time_dim)

        x = self.input_proj(x) + self.pos_embed[:, :L]
        x = self.encoder(x, src_key_padding_mask=mask)  # (B, L, hidden_dim)
        x = self.final_norm(x)

        if mask is not None:
            valid = (~mask).to(x.dtype).unsqueeze(-1)  # (B, L, 1)
            x = (x * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
        else:
            x = x.mean(dim=1)  # (B, hidden_dim)
        return x
