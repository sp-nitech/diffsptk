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
import torch.nn.functional as F

from ..utils.private import check_size, filter_values
from .base import BaseFunctionalModule, Precomputed
from .linear_intpl import LinearInterpolation
from .poledf import AllPoleDigitalFilter


class AllPoleLatticeDigitalFilter(BaseFunctionalModule):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/ltcdf.html>`_
    for details.

    Parameters
    ----------
    filter_order : int >= 0
        The order of the filter, :math:`M`.

    frame_period : int >= 1
        The frame period in samples, :math:`P`.

    ignore_gain : bool
        If True, perform filtering without the gain.

    References
    ----------
    .. [1] C.-Y. Yu et al., "Differentiable time-varying linear prediction in the
           context of end-to-end analysis-by-synthesis," *Proceedings of Interspeech*,
           2024.

    """

    _takes_input_size = True

    def __init__(
        self, filter_order: int, frame_period: int, ignore_gain: bool = False
    ) -> None:
        super().__init__()

        self.in_dim = filter_order + 1

        self._register_precomputed(self._precompute(**filter_values(locals())))

    def forward(self, x: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """Apply an all-pole lattice digital filter.

        Parameters
        ----------
        x : Tensor [shape=(..., T)]
            The excitation signal.

        k : Tensor [shape=(..., T/P, M+1)]
            The PARCOR coefficients.

        Returns
        -------
        out : Tensor [shape=(..., T)]
            The output signal.

        Examples
        --------
        >>> import diffsptk
        >>> ltcdf = diffsptk.AllPoleLatticeDigitalFilter(0, 1)
        >>> x = diffsptk.step(4)
        >>> k = diffsptk.ramp(4)
        >>> y = ltcdf(x, k.view(-1, 1))
        >>> y
        tensor([0., 1., 2., 3., 4.])

        """
        check_size(k.size(-1), self.in_dim, "dimension of PARCOR coefficients")
        return self._call_forward(x, k)

    @staticmethod
    def _func(x: torch.Tensor, k: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        _p = AllPoleLatticeDigitalFilter._precompute(k.size(-1) - 1, *args, **kwargs)
        return AllPoleLatticeDigitalFilter._apply_precomputed(_p, x=x, k=k)

    @staticmethod
    def _check(filter_order: int, frame_period: int) -> None:
        if filter_order < 0:
            raise ValueError("filter_order must be non-negative.")
        if frame_period <= 0:
            raise ValueError("frame_period must be positive.")

    @staticmethod
    def _precompute(
        filter_order: int, frame_period: int, ignore_gain: bool = False
    ) -> Precomputed:
        AllPoleLatticeDigitalFilter._check(filter_order, frame_period)
        return Precomputed(
            values={"frame_period": frame_period, "ignore_gain": ignore_gain}
        )

    @staticmethod
    def _forward(
        x: torch.Tensor,
        k: torch.Tensor,
        *,
        frame_period: int,
        ignore_gain: bool,
    ) -> torch.Tensor:
        check_size(x.size(-1), k.size(-2) * frame_period, "sequence length")

        k = LinearInterpolation._func(k, frame_period)
        K, k = torch.split(k, [1, k.size(-1) - 1], dim=-1)

        a = AllPoleLatticeDigitalFilter._lattice_to_direct_form(k)[..., 1:]
        a = torch.cat([K, a], dim=-1)
        y = AllPoleDigitalFilter._func(x, a, frame_period=1, ignore_gain=ignore_gain)
        return y

    @staticmethod
    def _lattice_to_direct_form(k: torch.Tensor) -> torch.Tensor:
        # Convert the time-varying lattice filter into the equivalent time-varying
        # direct-form filter. The backward polynomial is delayed by one sample
        # because the lattice states are computed with the previous coefficients.
        A = k.new_ones(*k.shape[:-1], 1)
        B = k.new_ones(*k.shape[:-1], 1)
        for m in range(k.size(-1)):
            km = k[..., m : m + 1]
            A = F.pad(A, (0, 1))
            B = F.pad(B[..., :-1, :], (1, 0, 1, 0))
            A_prev = A
            A = A + km * B
            B = B + km * A_prev
        return A
