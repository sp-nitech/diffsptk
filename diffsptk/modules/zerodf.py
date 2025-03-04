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

import torch.nn.functional as F

from ..utils.private import check_size
from ..utils.private import get_values
from .base import BaseFunctionalModule
from .linear_intpl import LinearInterpolation


class AllZeroDigitalFilter(BaseFunctionalModule):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/zerodf.html>`_
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

    def __init__(self, filter_order, frame_period, ignore_gain=False):
        super().__init__()

        self.in_dim = filter_order + 1

        self.values = self._precompute(*get_values(locals()))

    def forward(self, x, b):
        """Apply an all-zero digital filter.

        Parameters
        ----------
        x : Tensor [shape=(..., T)]
            The excitation signal.

        b : Tensor [shape=(..., T/P, M+1)]
            The filter coefficients.

        Returns
        -------
        out : Tensor [shape=(..., T)]
            The output signal.

        Examples
        --------
        >>> x = diffsptk.step(4)
        >>> b = diffsptk.ramp(4)
        >>> zerodf = diffsptk.AllZeroDigitalFilter(0, 1)
        >>> y = zerodf(x, b.view(-1, 1))
        >>> y
        tensor([[0., 1., 2., 3., 4.]])

        """
        check_size(b.size(-1), self.in_dim, "dimension of impulse response")
        return self._forward(x, b, *self.values)

    @staticmethod
    def _takes_input_size():
        return True

    @staticmethod
    def _func(x, b, *args, **kwargs):
        values = AllZeroDigitalFilter._precompute(b.size(-1) - 1, *args, **kwargs)
        return AllZeroDigitalFilter._forward(x, b, *values)

    @staticmethod
    def _check(filter_order, frame_period):
        if filter_order < 0:
            raise ValueError("filter_order must be non-negative.")
        if frame_period <= 0:
            raise ValueError("frame_period must be positive.")

    @staticmethod
    def _precompute(filter_order, frame_period, ignore_gain=False):
        AllZeroDigitalFilter._check(filter_order, frame_period)
        return (frame_period, ignore_gain)

    @staticmethod
    def _forward(x, b, frame_period, ignore_gain):
        check_size(x.size(-1), b.size(-2) * frame_period, "sequence length")

        M = b.size(-1) - 1
        x = F.pad(x, (M, 0))
        x = x.unfold(-1, M + 1, 1)
        h = LinearInterpolation._func(b.flip(-1), frame_period)
        if ignore_gain:
            h = h / h[..., -1:]
        y = (x * h).sum(-1)
        return y
