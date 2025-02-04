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

from ..misc.utils import check_size
from ..misc.utils import to
from .freqt2 import SecondOrderAllPassFrequencyTransform


class SecondOrderAllPassInverseFrequencyTransform(nn.Module):
    """Second-order all-pass inverse frequency transform module.

    Parameters
    ----------
    in_order : int >= 0
        Order of input sequence, :math:`M_1`.

    out_order : int >= 0
        Order of output sequence, :math:`M_2`.

    alpha : float in (-1, 1)
        Frequency warping factor, :math:`\\alpha`.

    theta : float in [0, 1]
        Emphasis frequency, :math:`\\theta`.

    n_fft : int >> M1
        Number of FFT bins. Accurate conversion requires the large value.

    References
    ----------
    .. [1] T. Wakako et al., "Speech spectral estimation based on expansion of log
           spectrum by arbitrary basis functions," *IEICE Trans*, vol. J82-D-II, no. 12,
           pp. 2203-2211, 1999 (in Japanese).

    """

    def __init__(self, in_order, out_order, alpha=0, theta=0, n_fft=512):
        super().__init__()

        assert 0 <= in_order < n_fft
        assert 0 <= out_order
        assert abs(alpha) < 1
        assert 0 <= theta <= 1

        self.in_order = in_order
        self.out_order = out_order
        self.register_buffer(
            "A", self._precompute(self.in_order, self.out_order, alpha, theta, n_fft)
        )

    def forward(self, c):
        """Perform second-order all-pass inverse frequency transform.

        Parameters
        ----------
        c : Tensor [shape=(..., M1+1)]
            Warped sequence.

        Returns
        -------
        out : Tensor [shape=(..., M2+1)]
            Output sequence.

        Examples
        --------
        >>> c1 = diffsptk.nrand(3)
        >>> c1
        tensor([ 0.0304,  0.5849, -0.8668, -0.7278])
        >>> freqt2 = diffsptk.SecondOrderAllPassFrequencyTransform(3, 4, .1, .3)
        >>> c2 = freqt2(c1)
        >>> c2
        tensor([ 0.0682,  0.4790, -1.0168, -0.6026,  0.1094])
        >>> ifreqt2 = diffsptk.SecondOrderAllPassInverseFrequencyTransform(4, 3, .1, .3)
        >>> c3 = ifreqt2(c2)
        >>> c3
        tensor([ 0.0682,  0.4790, -1.0168, -0.6026,  0.1094])

        """
        check_size(c.size(-1), self.in_order + 1, "dimension of cepstrum")
        return self._forward(c, self.A)

    @staticmethod
    def _forward(c, A):
        return torch.matmul(c, A)

    @staticmethod
    def _func(c, out_order, alpha, theta, n_fft):
        in_order = c.size(-1) - 1
        A = SecondOrderAllPassInverseFrequencyTransform._precompute(
            in_order, out_order, alpha, theta, n_fft, dtype=c.dtype, device=c.device
        )
        return SecondOrderAllPassInverseFrequencyTransform._forward(c, A)

    @staticmethod
    def _precompute(in_order, out_order, alpha, theta, n_fft, dtype=None, device=None):
        theta *= torch.pi

        k = torch.arange(n_fft, dtype=torch.double, device=device)
        omega = k * (2 * torch.pi / n_fft)
        ww = SecondOrderAllPassFrequencyTransform.warp(omega, alpha, theta)

        m1 = torch.arange(-in_order, in_order + 1, dtype=torch.double, device=device)
        wwm1 = ww.reshape(-1, 1) * m1.reshape(1, -1)
        real = torch.cos(wwm1)
        imag = -torch.sin(wwm1)

        A = torch.fft.ifft(torch.complex(real, imag), dim=0).real
        L = out_order + 1
        M = in_order + 1
        A[:L, M:] += A[:L, : (M - 1)].flip(1)
        A = A[:L, (M - 1) :]
        A[1:, 0] *= 2
        A[0, 1:] /= 2
        return to(A.T, dtype=dtype)
