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
import torch.nn as nn
from torchlpc import sample_wise_lpc

from ..misc.utils import check_size
from .linear_intpl import LinearInterpolation


class AllPoleDigitalFilter(nn.Module):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/poledf.html>`_
    for details.

    Parameters
    ----------
    filter_order : int >= 0 [scalar]
        Order of filter coefficients, :math:`M`.

    frame_period : int >= 1 [scalar]
        Frame period, :math:`P`.

    ignore_gain : bool [scalar]
        If True, perform filtering without gain.

    """

    def __init__(self, filter_order, frame_period, ignore_gain=False):
        super(AllPoleDigitalFilter, self).__init__()

        self.filter_order = filter_order
        self.frame_period = frame_period
        self.ignore_gain = ignore_gain

        assert 0 <= self.filter_order

        self.linear_intpl = LinearInterpolation(self.frame_period)

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
        y : Tensor [shape=(..., T)]
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

        d = x.dim()
        if d == 1:
            a = a.unsqueeze(0)
            x = x.unsqueeze(0)

        a = self.linear_intpl(a)
        K, a = torch.split(a, [1, self.filter_order], dim=-1)
        if not self.ignore_gain:
            x = K[..., 0] * x

        y = sample_wise_lpc(x, a)

        if d == 1:
            y = y.squeeze(0)
        return y
