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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..misc.utils import numpy_to_torch


class GroupDelay(nn.Module):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/grpdelay.html>`_
    for details.

    Parameters
    ----------
    fft_length : int >= 2 [scalar]
        Number of FFT bins, :math:`L`.

    alpha : float > 0 [scalar]
        Tuning parameter, :math:`\\alpha`.

    gamma : float > 0 [scalar]
        Tuning parameter, :math:`\\gamma`.

    """

    def __init__(self, fft_length, alpha=1, gamma=1):
        super(GroupDelay, self).__init__()

        self.fft_length = fft_length
        self.alpha = alpha
        self.gamma = gamma

        assert 2 <= self.fft_length
        assert 0 < self.alpha
        assert 0 < self.gamma

        ramp = np.arange(self.fft_length)
        self.register_buffer("ramp", numpy_to_torch(ramp))

    def forward(self, b, a=None):
        """Compute group delay.

        Parameters
        ----------
        b : Tensor [shape=(..., M+1)]
            Numerator coefficients.

        a : Tensor [shape=(..., N+1)]
            Denominator coefficients.

        Returns
        -------
        g : Tensor [shape=(..., L/2+1)]
            Group delay or modified group delay function.

        Examples
        --------
        >>> x = diffsptk.ramp(3)
        >>> grpdelay = diffsptk.GroupDelay(8)
        >>> g = grpdelay(x)
        >>> g
        tensor([2.3333, 2.4278, 3.0000, 3.9252, 3.0000])

        """
        if a is None:
            order = 0
            c = b
        else:
            order = a.size(-1) - 1

            # Remove gain.
            K, a1 = torch.split(a, [1, order], dim=-1)
            a2 = F.pad(a1, (1, 0), value=1).unsqueeze(-1)

            # Perform full convolution.
            b1 = F.pad(b, (order, order))
            b2 = b1.unfold(-1, b.size(-1) + order, 1)
            c = (b2 * a2).sum(-2)

        length = c.size(-1)
        assert length <= self.fft_length, "Please increase FFT length"

        d = c * self.ramp[:length]
        C = torch.fft.rfft(c, n=self.fft_length)
        D = torch.fft.rfft(d, n=self.fft_length)

        denom = C.real * C.real + C.imag * C.imag
        if self.gamma != 1:
            denom = torch.pow(denom, self.gamma)
        numer = C.real * D.real + C.imag * D.imag

        g = numer / denom - order
        if self.alpha != 1:
            g = torch.sign(g) * torch.pow(torch.abs(g), self.alpha)

        return g
