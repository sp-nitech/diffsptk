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

from ..utils.private import check_size, filter_values
from .base import BaseFunctionalModule, Precomputed
from .linear_intpl import LinearInterpolation
from .ltcdf import AllPoleLatticeDigitalFilter
from .zerodf import AllZeroDigitalFilter


class AllZeroLatticeDigitalFilter(BaseFunctionalModule):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/iltcdf.html>`_
    for details.

    Parameters
    ----------
    filter_order : int >= 0
        The order of the filter, :math:`M`.

    frame_period : int >= 1
        The frame period in samples, :math:`P`.

    ignore_gain : bool
        If True, perform filtering without the gain.

    """

    _takes_input_size = True

    def __init__(
        self, filter_order: int, frame_period: int, ignore_gain: bool = False
    ) -> None:
        super().__init__()

        self.in_dim = filter_order + 1

        self._register_precomputed(self._precompute(**filter_values(locals())))

    def forward(self, x: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """Apply an all-zero lattice digital filter.

        Parameters
        ----------
        x : Tensor [shape=(..., T)]
            The input signal.

        k : Tensor [shape=(..., T/P, M+1)]
            The PARCOR coefficients.

        Returns
        -------
        out : Tensor [shape=(..., T)]
            The residual signal.

        Examples
        --------
        >>> import diffsptk
        >>> iltcdf = diffsptk.AllZeroLatticeDigitalFilter(0, 1)
        >>> x = diffsptk.step(4)
        >>> k = diffsptk.ramp(1, 5)
        >>> y = iltcdf(x, k.view(-1, 1))
        >>> y
        tensor([1.0000, 0.5000, 0.3333, 0.2500, 0.2000])

        """
        check_size(k.size(-1), self.in_dim, "dimension of PARCOR coefficients")
        return self._call_forward(x, k)

    @staticmethod
    def _func(x: torch.Tensor, k: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        _p = AllZeroLatticeDigitalFilter._precompute(k.size(-1) - 1, *args, **kwargs)
        return AllZeroLatticeDigitalFilter._apply_precomputed(_p, x=x, k=k)

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
        AllZeroLatticeDigitalFilter._check(filter_order, frame_period)
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

        a = AllPoleLatticeDigitalFilter._lattice_to_direct_form(k)
        y = AllZeroDigitalFilter._func(x, a, frame_period=1)
        if not ignore_gain:
            y = y / K[..., 0]
        return y
