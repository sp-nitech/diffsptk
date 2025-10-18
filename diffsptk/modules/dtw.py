# ------------------------------------------------------------------------ #
# Copyright 2022 SPTK Working Group                                        #
#                                                                          #
# Licensed under the Apache License, Version 2.0 (the "License");          #
# you may not use this file except in compliance with the License.         #
# You may obtain a copy of the License at                                  #
#                                                                          #
#     http://www.apache.org/licenses/LICENSE-2.0                           #
#                                                                          #
# Unless required by applicable law or agreed to in writing, software      #
# distributed under the License is distributed on an "AS IS" BASIS,        #
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. #
# See the License for the specific language governing permissions and      #
# limitations under the License.                                           #
# ------------------------------------------------------------------------ #

import torch

from ..typing import Callable, Precomputed
from ..utils.private import filter_values
from .base import BaseFunctionalModule


def _soft_dtw_core(
    D: torch.Tensor,
    lengths: torch.Tensor,
    return_indices: bool,
    steps: list[tuple[int]],
    has_two_step_transition: bool,
    gamma: float,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    B, T1, T2 = D.shape

    R = torch.full_like(D, float("inf"))
    R_ = R.clone() if has_two_step_transition else None
    R[:, 0, 0] = D[:, 0, 0]

    if return_indices:
        P = torch.full((B, T1, T2, 2), -1, device=D.device, dtype=torch.long)
        P_ = P.clone() if has_two_step_transition else None

    # Forward computation.
    for i in range(T1):
        for j in range(T2):
            rs = []
            rs_ = []
            if return_indices:
                ps = []
                ps_ = []

            d = D[:, i, j]
            for k in range(len(steps)):
                i_k = i - steps[k][0]
                j_k = j - steps[k][1]
                if i_k < 0 or j_k < 0:
                    continue

                if return_indices:
                    p = torch.tensor([i_k, j_k], device=D.device, dtype=torch.long)
                w = sum(steps[k])

                if has_two_step_transition:
                    if steps[k][0] == 0 or steps[k][1] == 0:
                        if R_[:, i_k, j_k] != float("inf"):
                            rs.append(d * w + R_[:, i_k, j_k])
                            if return_indices:
                                ps.append(p)
                    else:
                        if R[:, i_k, j_k] != float("inf"):
                            rs.append(d * w + R[:, i_k, j_k])
                            rs_.append(rs[-1])
                            if return_indices:
                                ps.append(p)
                                ps_.append(ps[-1])
                else:
                    if R[:, i_k, j_k] != float("inf"):
                        rs.append(d * w + R[:, i_k, j_k])
                        if return_indices:
                            ps.append(p)

            if 0 < len(rs):
                rs = torch.stack(rs, dim=0)
                r = -gamma * torch.logsumexp(-rs / gamma, dim=0)
                R[:, i, j] = r

                if return_indices:
                    ps = torch.stack(ps, dim=0)  # (K, 2)
                    p = torch.argmin(rs, dim=0)  # (B,)
                    P[:, i, j] = ps[p]

            if 0 < len(rs_):
                rs_ = torch.stack(rs_, dim=0)
                r_ = -gamma * torch.logsumexp(-rs_ / gamma, dim=0)
                R_[:, i, j] = r_

                if return_indices:
                    ps_ = torch.stack(ps_, dim=0)
                    p_ = torch.argmin(rs_, dim=0)
                    P_[:, i, j] = ps_[p_]

    distance = R[torch.arange(B, device=D.device), lengths[:, 0] - 1, lengths[:, 1] - 1]
    distance = distance / lengths.sum(dim=1).to(distance.dtype)

    # Backtracking.
    viterbi_paths = []
    if return_indices:
        for b in range(B):
            two_step_transition = False
            ij = lengths[b] - 1
            viterbi_path = [ij]
            while (0 <= ij).all():
                if has_two_step_transition and two_step_transition:
                    prev_ij = P_[b, ij[0], ij[1]]
                else:
                    prev_ij = P[b, ij[0], ij[1]]
                if (0 <= prev_ij).all():
                    viterbi_path.append(prev_ij)
                two_step_transition = (prev_ij == ij).any()
                ij = prev_ij

            viterbi_paths.append(torch.stack(viterbi_path[::-1], dim=0))

    return distance, viterbi_paths


class DynamicTimeWarping(BaseFunctionalModule):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/dtw.html>`_
    for details. The current implementation is based on naive nested for loops.

    Parameters
    ----------
    metric : ['manhattan', 'euclidean', 'squared-euclidean', 'symmetric-kl']
        The metric to compute the distance between two vectors.

    p : int in [0, 6]
        The local path constraint type.

    softness : float > 0
        A smoothing parameter. The smaller value makes the output closer to the true
        dynamic time warping distance, but the gradient vanishes.

    References
    ----------
    .. [1] M. Cuturi et al., "Soft-DTW: a differentiable loss function for time-series,"
           *Proceedings of ICML 2017*, pp. 894-903, 2017.

    """

    def __init__(
        self,
        metric: str | int = "euclidean",
        p: int = 4,
        softness: float = 1e-3,
    ) -> None:
        super().__init__()

        self.values = self._precompute(**filter_values(locals()))

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        lengths: torch.Tensor | None = None,
        return_indices: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """Compute dynamic time warping distance.

        Parameters
        ----------
        x : Tensor [shape=(B, T1, D) or (T1, D) or (T1,)]
            The query vector sequence.

        y : Tensor [shape=(B, T2, D) or (T2, D) or (T2,)]
            The reference vector sequence.

        lengths : Tensor [shape=(B, 2)] or None
            The lengths of the sequences.

        return_indices : bool
            If True, returns the indices of the viterbi path.

        Returns
        -------
        distance : Tensor [shape=(B,)]
            The dynamic time warping distance.

        indices : list[Tensor [shape=(T, 2)]] (optional)
            The indices of the viterbi path for each batch.

        Examples
        --------
        >>> import diffsptk
        >>> dtw = diffsptk.DynamicTimeWarping(p=1)
        >>> x = torch.tensor([1., 3., 6., 9.])
        >>> y = torch.tensor([2., 3., 8., 8.])
        >>> distance, indices = dtw(x, y, return_indices=True)
        >>> distance
        tensor([0.8749])
        >>> indices[0]
        tensor([[0, 0],
                [1, 1],
                [2, 2],
                [3, 2],
                [3, 3]])

        """
        return self._forward(x, y, lengths, return_indices, *self.values)

    @staticmethod
    def _func(
        x: torch.Tensor,
        y: torch.Tensor,
        lengths: torch.Tensor | None,
        return_indices: bool,
        *args,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        values = DynamicTimeWarping._precompute(*args, **kwargs)
        return DynamicTimeWarping._forward(x, y, lengths, return_indices, *values)

    @staticmethod
    def _takes_input_size() -> bool:
        return False

    @staticmethod
    def _check(softness: float) -> None:
        if softness <= 0:
            raise ValueError("softness must be positive.")

    @staticmethod
    def _precompute(metric: str | int, p: int, softness: float) -> Precomputed:
        DynamicTimeWarping._check(softness)

        if metric in (0, "manhattan"):
            dist_func = lambda x, y: torch.cdist(x, y, p=1)
        elif metric in (1, "euclidean"):
            dist_func = lambda x, y: torch.cdist(x, y, p=2)
        elif metric in (2, "squared-euclidean"):
            dist_func = lambda x, y: torch.cdist(x, y, p=2).square()
        elif metric in (3, "symmetric-kl"):

            def sym_kl(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                x = x.unsqueeze(-2)
                y = y.unsqueeze(-3)
                kl1 = (x * (x / y).clip(min=1e-10).log()).sum(-1)
                kl2 = (y * (y / x).clip(min=1e-10).log()).sum(-1)
                return kl1 + kl2

            dist_func = sym_kl
        else:
            raise ValueError(f"metric {metric} is not supported.")

        dtw_func = _soft_dtw_core

        local_path_constraints = {
            0: {
                "steps": [(1, 0), (0, 1)],
                "has_two_step_transition": False,
            },
            1: {
                "steps": [(1, 0), (0, 1), (1, 1)],
                "has_two_step_transition": False,
            },
            2: {
                "steps": [(1, 0), (1, 1)],
                "has_two_step_transition": False,
            },
            3: {
                "steps": [(1, 0), (1, 1), (1, 2)],
                "has_two_step_transition": False,
            },
            4: {
                "steps": [(1, 0), (0, 1), (1, 1)],
                "has_two_step_transition": True,
            },
            5: {
                "steps": [(1, 1), (1, 2), (2, 1)],
                "has_two_step_transition": False,
            },
            6: {
                "steps": [(1, 0), (1, 1), (1, 2)],
                "has_two_step_transition": True,
            },
        }
        if p not in local_path_constraints:
            raise ValueError(f"local path constraint type {p} is not supported.")
        steps = local_path_constraints[p]["steps"]
        has_two_step_transition = local_path_constraints[p]["has_two_step_transition"]

        return (steps, has_two_step_transition, softness, dist_func, dtw_func)

    @staticmethod
    def _forward(
        x: torch.Tensor,
        y: torch.Tensor,
        lengths: torch.Tensor | None,
        return_indices: bool,
        steps: list[tuple[int]],
        has_two_step_transition: bool,
        softness: float,
        dist_func: Callable,
        dtw_func: Callable,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        if x.dim() == 1:
            x = x.view(1, -1, 1)
            y = y.view(1, -1, 1)
        elif x.dim() == 2:
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)
        if x.dim() != 3:
            raise ValueError("x and y must be 1D, 2D, or 3D tensor.")
        if x.dim() != y.dim():
            raise ValueError("x and y must have the same number of dimensions.")

        D = dist_func(x, y)

        if lengths is None:
            B, T1, T2 = D.shape
            lengths = torch.tensor(B * [[T1, T2]], device=x.device, dtype=torch.long)
        if lengths.dim() != 2:
            raise ValueError("lengths must be 2D tensor.")

        distance, indices = dtw_func(
            D, lengths, return_indices, steps, has_two_step_transition, softness
        )

        if return_indices:
            return distance, indices
        return distance

    @staticmethod
    def merge(
        x: torch.Tensor,
        y: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        """Merge two sequences according to the given indices.

        Parameters
        ----------
        x : Tensor [shape=(T1, D) or (T1,)]
            The query vector sequence.

        y : Tensor [shape=(T2, D) or (T2,)]
            The reference vector sequence.

        indices : Tensor [shape=(T, 2)]
            The indices of the viterbi path.

        Returns
        -------
        z : Tensor [shape=(T, 2D) or (T, 2)]
            The merged vector sequence.

        Examples
        --------
        >>> import diffsptk
        >>> dtw = diffsptk.DynamicTimeWarping(p=1)
        >>> x = torch.tensor([1., 3., 6., 9.])
        >>> y = torch.tensor([2., 3., 8., 8.])
        >>> _, indices = dtw(x, y, return_indices=True)
        >>> z = dtw.merge(x, y, indices[0])
        >>> z
        tensor([[1., 2.],
                [3., 3.],
                [6., 8.],
                [9., 8.],
                [9., 8.]])

        """
        if x.dim() != y.dim():
            raise ValueError("x and y must have the same number of dimensions.")
        if indices.dim() != 2 or indices.size(-1) != 2:
            raise ValueError("The shape of indices must be (T, 2).")
        x_expanded = x[indices[:, 0]]
        y_expanded = y[indices[:, 1]]
        if x.dim() == 1:
            z = torch.stack([x_expanded, y_expanded], dim=-1)
        else:
            z = torch.cat([x_expanded, y_expanded], dim=-1)
        return z
