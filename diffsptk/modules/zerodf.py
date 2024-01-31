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

import torch.nn as nn

from ..misc.utils import check_size
from .linear_intpl import LinearInterpolation


class AllZeroDigitalFilter(nn.Module):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/zerodf.html>`_
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
        super(AllZeroDigitalFilter, self).__init__()

        self.filter_order = filter_order
        self.frame_period = frame_period
        self.ignore_gain = ignore_gain

        assert 0 <= self.filter_order

        self.pad = nn.ConstantPad1d((self.filter_order, 0), 0)
        self.linear_intpl = LinearInterpolation(self.frame_period)

    def forward(self, x, h):
        """Apply an all-zero digital filter.

        Parameters
        ----------
        x : Tensor [shape=(..., T)]
            Excitation signal.

        h : Tensor [shape=(..., T/P, M+1)]
            Filter coefficients.

        Returns
        -------
        y : Tensor [shape=(..., T)]
            Output signal.

        Examples
        --------
        >>> x = diffsptk.step(4)
        >>> h = diffsptk.ramp(4)
        >>> zerodf = diffsptk.AllZeroDigitalFilter(0, 1)
        >>> y = zerodf(x, h.view(-1, 1))
        >>> y
        tensor([[0., 1., 2., 3., 4.]])

        """
        check_size(h.size(-1), self.filter_order + 1, "dimension of impulse response")
        check_size(x.size(-1), h.size(-2) * self.frame_period, "sequence length")

        x = self.pad(x)
        x = x.unfold(-1, self.filter_order + 1, 1)
        h = self.linear_intpl(h.flip(-1))
        if self.ignore_gain:
            h = h / h[..., -1:]
        y = (x * h).sum(-1)
        return y
