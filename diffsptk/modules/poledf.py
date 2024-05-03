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
from torch import nn
from torchlpc import sample_wise_lpc

from ..misc.utils import check_size
from .linear_intpl import LinearInterpolation


class AllPoleDigitalFilter(nn.Module):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/poledf.html>`_
    for details.

    Parameters
    ----------
    filter_order : int >= 0
        Order of filter coefficients, :math:`M`.

    frame_period : int >= 1
        Frame period, :math:`P`.

    ignore_gain : bool
        If True, perform filtering without gain.

    """

    def __init__(self, filter_order, frame_period, ignore_gain=False):
        super().__init__()

        assert 0 <= filter_order
        assert 1 <= frame_period

        self.filter_order = filter_order
        self.frame_period = frame_period
        self.ignore_gain = ignore_gain

    def forward(self, x, a):
        """Apply an all-pole digital filter.

        Parameters
        ----------
        x : Tensor [shape=(..., T)]
            Excitation signal.

        a : Tensor [shape=(..., T/P, M+1)]
            Filter coefficients.

        Returns
        -------
        out : Tensor [shape=(..., T)]
            Output signal.

        Examples
        --------
        >>> x = diffsptk.step(4)
        >>> a = diffsptk.ramp(4)
        >>> poledf = diffsptk.AllPoleDigitalFilter(0, 1)
        >>> y = poledf(x, a.view(-1, 1))
        >>> y
        tensor([[0., 1., 2., 3., 4.]])

        """
        check_size(a.size(-1), self.filter_order + 1, "dimension of LPC coefficients")
        check_size(x.size(-1), a.size(-2) * self.frame_period, "sequence length")
        return self._forward(x, a, self.frame_period, self.ignore_gain)

    @staticmethod
    def _forward(x, a, frame_period, ignore_gain):
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

    _func = _forward
