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
from torch import nn

from ..misc.utils import remove_gain
from ..misc.utils import to


class GroupDelay(nn.Module):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/grpdelay.html>`_
    for details.

    Parameters
    ----------
    fft_length : int >= 2
        Number of FFT bins, :math:`L`.

    alpha : float > 0
        Tuning parameter, :math:`\\alpha`.

    gamma : float > 0
        Tuning parameter, :math:`\\gamma`.

    """

    def __init__(self, fft_length, alpha=1, gamma=1):
        super().__init__()

        assert 2 <= fft_length
        assert 0 < alpha
        assert 0 < gamma

        self.fft_length = fft_length
        self.alpha = alpha
        self.gamma = gamma
        self.register_buffer("ramp", self._precompute(self.fft_length))

    def forward(self, b=None, a=None):
        """Compute group delay.

        Parameters
        ----------
        b : Tensor [shape=(..., M+1)] or None
            Numerator coefficients.

        a : Tensor [shape=(..., N+1)] or None
            Denominator coefficients.

        Returns
        -------
        out : Tensor [shape=(..., L/2+1)]
            Group delay or modified group delay function.

        Examples
        --------
        >>> x = diffsptk.ramp(3)
        >>> grpdelay = diffsptk.GroupDelay(8)
        >>> g = grpdelay(x)
        >>> g
        tensor([2.3333, 2.4278, 3.0000, 3.9252, 3.0000])

        """
        return self._forward(
            b,
            a,
            self.fft_length,
            self.alpha,
            self.gamma,
            self.ramp,
        )

    @staticmethod
    def _forward(b, a, fft_length, alpha, gamma, ramp):
        if b is None and a is None:
            raise ValueError("Either b or a must be specified.")

        if a is None:
            order = 0
        else:
            a = remove_gain(a)
            order = a.size(-1) - 1

        if b is None:
            c = a.flip(-1)
        elif a is None:
            c = b
        else:
            # Perform full convolution.
            b1 = F.pad(b, (order, order))
            b2 = b1.unfold(-1, b.size(-1) + order, 1)
            c = (b2 * a.unsqueeze(-1)).sum(-2)

        data_length = c.size(-1)
        if fft_length < data_length:
            raise RuntimeError("Please increase FFT length.")

        d = c * ramp[:data_length]
        C = torch.fft.rfft(c, n=fft_length)
        D = torch.fft.rfft(d, n=fft_length)

        numer = C.real * D.real + C.imag * D.imag
        denom = C.real * C.real + C.imag * C.imag
        if gamma != 1:
            denom = torch.pow(denom, gamma)

        g = numer / denom - order
        if alpha != 1:
            g = torch.sign(g) * torch.pow(torch.abs(g), alpha)
        return g

    @staticmethod
    def _func(b, a, fft_length, alpha, gamma):
        if b is not None and a is not None:
            data_length = b.size(-1) + a.size(-1) - 1
        elif b is not None:
            data_length = b.size(-1)
        else:
            data_length = a.size(-1)
        ramp = GroupDelay._precompute(
            data_length,
            dtype=a.dtype if b is None else b.dtype,
            device=a.device if b is None else b.device,
        )
        return GroupDelay._forward(b, a, fft_length, alpha, gamma, ramp)

    @staticmethod
    def _precompute(length, dtype=None, device=None):
        ramp = torch.arange(length, device=device)
        return to(ramp, dtype=dtype)
