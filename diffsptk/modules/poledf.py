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
from torchlpc import sample_wise_lpc

from ..misc.utils import check_size
from ..misc.utils import get_values
from .base import BaseFunctionalModule
from .linear_intpl import LinearInterpolation


class AllPoleDigitalFilter(BaseFunctionalModule):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/poledf.html>`_
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

    def __init__(self, filter_order, frame_period, ignore_gain=False):
        super().__init__()

        self.in_dim = filter_order + 1

        self.values = self._precompute(*get_values(locals()))

    def forward(self, x, a):
        """Apply an all-pole digital filter.

        Parameters
        ----------
        x : Tensor [shape=(..., T)]
            The excitation signal.

        a : Tensor [shape=(..., T/P, M+1)]
            The filter coefficients.

        Returns
        -------
        out : Tensor [shape=(..., T)]
            The output signal.

        Examples
        --------
        >>> x = diffsptk.step(4)
        >>> a = diffsptk.ramp(4)
        >>> poledf = diffsptk.AllPoleDigitalFilter(0, 1)
        >>> y = poledf(x, a.view(-1, 1))
        >>> y
        tensor([[0., 1., 2., 3., 4.]])

        """
        check_size(a.size(-1), self.in_dim, "dimension of LPC coefficients")
        return self._forward(x, a, *self.values)

    @staticmethod
    def _func(x, a, *args, **kwargs):
        values = AllPoleDigitalFilter._precompute(a.size(-1) - 1, *args, **kwargs)
        return AllPoleDigitalFilter._forward(x, a, *values)

    @staticmethod
    def _takes_input_size():
        return True

    @staticmethod
    def _check(filter_order, frame_period):
        if filter_order < 0:
            raise ValueError("filter_order must be non-negative.")
        if frame_period <= 0:
            raise ValueError("frame_period must be positive.")

    @staticmethod
    def _precompute(filter_order, frame_period, ignore_gain=False):
        AllPoleDigitalFilter._check(filter_order, frame_period)
        return (frame_period, ignore_gain)

    @staticmethod
    def _forward(x, a, frame_period, ignore_gain):
        check_size(x.size(-1), a.size(-2) * frame_period, "sequence length")

        d = x.dim()
        if d == 1:
            a = a.unsqueeze(0)
            x = x.unsqueeze(0)

        a = LinearInterpolation._func(a, frame_period)
        K, a = torch.split(a, [1, a.size(-1) - 1], dim=-1)
        if not ignore_gain:
            x = K[..., 0] * x

        y = sample_wise_lpc(x, a)
        if d == 1:
            y = y.squeeze(0)
        return y
