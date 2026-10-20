# Functional Tucker representation with SIREN coordinate-MLP bases.
# Adapted from SDIFT (Chen et al., 2025): https://github.com/OceanSTARLab/SDIFT
# Core component of DyPSI (NeurIPS 2026).

from typing import Optional, Tuple

import torch
from torch import nn


class SineLayer(nn.Module):
    """Fully connected layer with a sin(sin(.)) activation (SIREN-style unit).

    Args:
        omega_0: Frequency scaling applied to the linear output.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 omega_0: float = 1) -> None:
        super().__init__()
        self.omega_0 = omega_0
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()

    def init_weights(self) -> None:
        with torch.no_grad():
            self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """input: (..., in_features) -> (..., out_features)."""
        return torch.sin(torch.sin(self.omega_0 * self.linear(input)))


class Continuous_Tucker_ssf(nn.Module):
    """Continuous Tucker factor model with SIREN bases and a fixed core.

    The model value at a queried (t, y, x) index tuple is the Tucker
    contraction of the three factor rows with the flattened core.

    Args:
        r_1, r_2, r_3: Tucker ranks of the temporal and two spatial modes.
        core: Flattened core tensor of shape (r_1 * r_2 * r_3,).
    """

    def __init__(self, r_1: int, r_2: int, r_3: int, core: torch.Tensor) -> None:
        super(Continuous_Tucker_ssf, self).__init__()
        self.r_1 = r_1
        self.r_2 = r_2
        self.r_3 = r_3

        mid_channel = 512
        omega = 4
        self.U_net = nn.Sequential(SineLayer(1, mid_channel, omega_0=omega),
                                   SineLayer(mid_channel, mid_channel, omega_0=omega),
                                   nn.Linear(mid_channel, r_1))

        self.V_net = nn.Sequential(SineLayer(1, mid_channel, omega_0=omega),
                                   SineLayer(mid_channel, mid_channel, omega_0=omega),
                                   nn.Linear(mid_channel, r_2))

        self.W_net = nn.Sequential(SineLayer(1, mid_channel, omega_0=omega),
                                   SineLayer(mid_channel, mid_channel, omega_0=omega),
                                   nn.Linear(mid_channel, r_3))

        self.core = core

    def kronecker_product_einsum_batched(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """Batched Kronecker product: (b, a, c) x (b, k, p) -> (b, a*k, c*p)."""
        assert A.dim() == 3 and B.dim() == 3

        res = torch.einsum("bac,bkp->bakcp", A, B).view(A.size(0),
                                                        A.size(1) * B.size(1),
                                                        A.size(2) * B.size(2))
        return res

    def forward(self, train_ind_batch: torch.Tensor) -> torch.Tensor:
        """
        Args:
            train_ind_batch: Coordinate indices of shape (B, 3), columns are
                the (normalized) temporal and two spatial coordinates.

        Returns:
            Tensor of shape (B,) with the reconstructed scalar values.
        """
        U_input = train_ind_batch[:, 0].unsqueeze(1)
        V_input = train_ind_batch[:, 1].unsqueeze(1)
        W_input = train_ind_batch[:, 2].unsqueeze(1)

        U = self.U_net(U_input).unsqueeze(1)  # B * 1 * r_1
        V = self.V_net(V_input).unsqueeze(1)  # B * 1 * r_2
        W = self.W_net(W_input).unsqueeze(1)  # B * 1 * r_3

        UV = self.kronecker_product_einsum_batched(U, V)
        UVW = self.kronecker_product_einsum_batched(UV, W).squeeze(1)

        out_put = torch.einsum("bi, i->b", UVW, self.core)
        return out_put


class Tensor_inr_3D(nn.Module):
    """3D Tucker basis generator with SIREN coordinate-MLP factor networks.

    The ``mode`` property switches between training and sampling behavior:

    - ``training``: ``forward`` takes per-mode coordinate tensors and returns
      the factor rows (U, V, W).
    - ``sampling``: ``forward`` takes a (B, 3) coordinate tensor and returns
      the flattened Kronecker product rows of shape (B, r_1 * r_2 * r_3).

    Args:
        R: Tucker ranks (r_1, r_2, r_3).
        omega: SIREN frequency factor for the factor networks.
    """

    def __init__(self, R: Tuple[int, int, int], omega: float = 10) -> None:
        super(Tensor_inr_3D, self).__init__()
        self.r_1 = R[0]
        self.r_2 = R[1]
        self.r_3 = R[2]
        self._mode = "training"

        mid_channel = 1024

        self.U_net = nn.Sequential(SineLayer(1, mid_channel, omega_0=omega),
                                   SineLayer(mid_channel, mid_channel, omega_0=omega), nn.Dropout(0),
                                   nn.Linear(mid_channel, self.r_1), nn.Tanh())

        self.V_net = nn.Sequential(SineLayer(1, mid_channel, omega_0=omega),
                                   SineLayer(mid_channel, mid_channel, omega_0=omega), nn.Dropout(0),
                                   nn.Linear(mid_channel, self.r_2), nn.Tanh())

        self.W_net = nn.Sequential(SineLayer(1, mid_channel, omega_0=omega),
                                   SineLayer(mid_channel, mid_channel, omega_0=omega), nn.Dropout(0),
                                   nn.Linear(mid_channel, self.r_3), nn.Tanh())

    @property
    def mode(self) -> str:
        return self._mode

    @mode.setter
    def mode(self, mode: str) -> None:
        if mode not in ["training", "sampling"]:
            raise ValueError("Mode should be 'training' or 'sampling'")
        self._mode = mode

    def kronecker_product_einsum_batched(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """Batched Kronecker product: (b, a, c) x (b, k, p) -> (b, a*k, c*p)."""
        assert A.dim() == 3 and B.dim() == 3

        res = torch.einsum("bac,bkp->bakcp", A, B).view(A.size(0),
                                                        A.size(1) * B.size(1),
                                                        A.size(2) * B.size(2))
        return res

    def forward(self,
                input_ind_train: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
                input_ind_sampl: Optional[torch.Tensor] = None):
        """
        Args:
            input_ind_train: Tuple (U_ind, V_ind, W_ind) of coordinate tensors,
                each of shape (B,); used in ``training`` mode.
            input_ind_sampl: Coordinate tensor of shape (B, 3); used in
                ``sampling`` mode.

        Returns:
            ``training`` mode: tuple (U, V, W) with shapes (B, r_1), (B, r_2),
            (B, r_3). ``sampling`` mode: tensor of shape (B, r_1*r_2*r_3).
        """
        if self._mode == "training":
            U = self.U_net(input_ind_train[0].unsqueeze(1))  # B * r_1
            V = self.V_net(input_ind_train[1].unsqueeze(1))  # B * r_2
            W = self.W_net(input_ind_train[2].unsqueeze(1))  # B * r_3
            return (U, V, W)
        elif self._mode == "sampling":
            # input_ind_sampl: B * 3
            U = self.U_net(input_ind_sampl[:, :1]).unsqueeze(1)  # B * 1 * r_1
            V = self.V_net(input_ind_sampl[:, 1:2]).unsqueeze(1)  # B * 1 * r_2
            W = self.W_net(input_ind_sampl[:, 2:3]).unsqueeze(1)  # B * 1 * r_3
            UV = self.kronecker_product_einsum_batched(U, V)
            UVW = self.kronecker_product_einsum_batched(UV, W).squeeze(1)
            return UVW


class Tensor_inr_4D(nn.Module):
    """4D Tucker basis generator with SIREN coordinate-MLP factor networks.

    The ``mode`` property switches between training and sampling behavior:

    - ``training``: ``forward`` takes per-mode coordinate tensors and returns
      the factor rows (T, U, V, W).
    - ``sampling``: ``forward`` takes a (B, 4) coordinate tensor and returns
      the flattened Kronecker product rows of shape
      (B, r_1 * r_2 * r_3 * r_4).

    Args:
        R: Tucker ranks (r_1, r_2, r_3, r_4).
        omega: SIREN frequency factor for the factor networks.
    """

    def __init__(self, R: Tuple[int, int, int, int], omega: float = 10) -> None:
        super(Tensor_inr_4D, self).__init__()
        self.r_1 = R[0]
        self.r_2 = R[1]
        self.r_3 = R[2]
        self.r_4 = R[3]
        self._mode = "training"

        mid_channel = 1024

        self.T_net = nn.Sequential(SineLayer(1, mid_channel, omega_0=omega),
                                   SineLayer(mid_channel, mid_channel, omega_0=omega), nn.Dropout(0),
                                   nn.Linear(mid_channel, self.r_1), nn.Tanh())

        self.U_net = nn.Sequential(SineLayer(1, mid_channel, omega_0=omega),
                                   SineLayer(mid_channel, mid_channel, omega_0=omega), nn.Dropout(0),
                                   nn.Linear(mid_channel, self.r_2), nn.Tanh())

        self.V_net = nn.Sequential(SineLayer(1, mid_channel, omega_0=omega),
                                   SineLayer(mid_channel, mid_channel, omega_0=omega), nn.Dropout(0),
                                   nn.Linear(mid_channel, self.r_3), nn.Tanh())

        self.W_net = nn.Sequential(SineLayer(1, mid_channel, omega_0=omega),
                                   SineLayer(mid_channel, mid_channel, omega_0=omega), nn.Dropout(0),
                                   nn.Linear(mid_channel, self.r_4), nn.Tanh())

    @property
    def mode(self) -> str:
        return self._mode

    @mode.setter
    def mode(self, mode: str) -> None:
        if mode not in ["training", "sampling"]:
            raise ValueError("Mode should be 'training' or 'sampling'")
        self._mode = mode

    def kronecker_product_einsum_batched(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """Batched Kronecker product: (b, a, c) x (b, k, p) -> (b, a*k, c*p)."""
        assert A.dim() == 3 and B.dim() == 3

        res = torch.einsum("bac,bkp->bakcp", A, B).view(A.size(0),
                                                        A.size(1) * B.size(1),
                                                        A.size(2) * B.size(2))
        return res

    def forward(self,
                input_ind_train: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = None,
                input_ind_sampl: Optional[torch.Tensor] = None):
        """
        Args:
            input_ind_train: Tuple (T_ind, U_ind, V_ind, W_ind) of coordinate
                tensors, each of shape (B,); used in ``training`` mode.
            input_ind_sampl: Coordinate tensor of shape (B, 4), columns in
                (t, u, v, w) order; used in ``sampling`` mode.

        Returns:
            ``training`` mode: tuple (T, U, V, W) with shapes (B, r_1) ...
            (B, r_4). ``sampling`` mode: tensor of shape
            (B, r_1*r_2*r_3*r_4), T-major (C-order) flattening.
        """
        if self._mode == "training":
            T = self.T_net(input_ind_train[0].unsqueeze(1))
            U = self.U_net(input_ind_train[1].unsqueeze(1))  # B * r_2
            V = self.V_net(input_ind_train[2].unsqueeze(1))  # B * r_3
            W = self.W_net(input_ind_train[3].unsqueeze(1))  # B * r_4
            return (T, U, V, W)
        elif self._mode == "sampling":
            # input_ind_sampl: B * 4
            T = self.T_net(input_ind_sampl[:, :1]).unsqueeze(1)  # B * 1 * r_1
            U = self.U_net(input_ind_sampl[:, 1:2]).unsqueeze(1)  # B * 1 * r_2
            V = self.V_net(input_ind_sampl[:, 2:3]).unsqueeze(1)  # B * 1 * r_3
            W = self.W_net(input_ind_sampl[:, 3:4]).unsqueeze(1)  # B * 1 * r_4
            TU = self.kronecker_product_einsum_batched(T, U)
            TUV = self.kronecker_product_einsum_batched(TU, V)
            TUVW = self.kronecker_product_einsum_batched(TUV, W).squeeze(1)
            return TUVW


def tucker_reconstruct_3d(core: torch.Tensor, U: torch.Tensor, V: torch.Tensor,
                          W: torch.Tensor) -> torch.Tensor:
    """Reconstruct a spatio-temporal field from a Tucker core and factors.

    Args:
        core: Core tensor of shape (B, T, R1, R2, R3).
        U, V, W: Factor matrices of shapes (T, R1), (H, R2), (Wd, R3).

    Returns:
        Reconstructed field of shape (B, T, H, Wd).
    """
    out = torch.einsum('btijk,ti->btjk', core, U)
    out = torch.einsum('btjk,yj->btyk', out, V)
    out = torch.einsum('btyk,xk->btyx', out, W)
    return out


def tucker_reconstruct_4d(core: torch.Tensor, T: torch.Tensor, U: torch.Tensor,
                          V: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    """Reconstruct a 3D-spatial spatio-temporal field from a 4-mode Tucker core.

    Args:
        core: Core tensor of shape (B, T, R1, R2, R3, R4).
        T, U, V, W: Factor matrices of shapes (T, R1), (H, R2), (Wd, R3),
            (Z, R4).

    Returns:
        Reconstructed field of shape (B, T, H, Wd, Z).
    """
    out = torch.einsum('btijkl,ti->btjkl', core, T)
    out = torch.einsum('btjkl,yj->btykl', out, U)
    out = torch.einsum('btykl,xk->btyxl', out, V)
    out = torch.einsum('btyxl,zl->btyxz', out, W)
    return out
