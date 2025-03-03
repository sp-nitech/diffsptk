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

from ..misc.utils import check_size
from ..misc.utils import get_values
from ..misc.utils import to
from .base import BaseFunctionalModule


class SecondOrderAllPassFrequencyTransform(BaseFunctionalModule):
    """Second-order all-pass frequency transform module.

    Parameters
    ----------
    in_order : int >= 0
        The order of the input sequence, :math:`M_1`.

    out_order : int >= 0
        The order of the output sequence, :math:`M_2`.

    alpha : float in (-1, 1)
        The frequency warping factor, :math:`\\alpha`.

    theta : float in [0, 1]
        The emphasis frequency, :math:`\\theta`.

    n_fft : int >> M2
        The number of FFT bins. The accurate conversion requires the large value.

    References
    ----------
    .. [1] T. Wakako et al., "Speech spectral estimation based on expansion of log
           spectrum by arbitrary basis functions," *IEICE Trans*, vol. J82-D-II, no. 12,
           pp. 2203-2211, 1999 (in Japanese).

    """

    def __init__(self, in_order, out_order, alpha=0, theta=0, n_fft=512):
        super().__init__()

        self.in_dim = in_order + 1

        _, _, tensors = self._precompute(*get_values(locals()))
        self.register_buffer("A", tensors[0])

    def forward(self, c):
        """Perform second-order all-pass frequency transform.

        Parameters
        ----------
        c : Tensor [shape=(..., M1+1)]
            The input sequence.

        Returns
        -------
        out : Tensor [shape=(..., M2+1)]
            The warped sequence.

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
        check_size(c.size(-1), self.in_dim, "dimension of cepstrum")
        return self._forward(c, **self._buffers)

    @staticmethod
    def _func(c, *args, **kwargs):
        _, _, tensors = SecondOrderAllPassFrequencyTransform._precompute(
            c.size(-1) - 1, *args, **kwargs, device=c.device, dtype=c.dtype
        )
        return SecondOrderAllPassFrequencyTransform._forward(c, *tensors)

    @staticmethod
    def _takes_input_size():
        return True

    @staticmethod
    def _check(in_order, out_order, alpha, theta):
        if in_order < 0:
            raise ValueError("in_order must be non-negative.")
        if out_order < 0:
            raise ValueError("out_order must be non-negative.")
        if 1 <= abs(alpha):
            raise ValueError("alpha must be in (-1, 1).")
        if not 0 <= theta <= 1:
            raise ValueError("theta must be in [0, 1].")

    @staticmethod
    def _precompute(in_order, out_order, alpha, theta, n_fft, device=None, dtype=None):
        SecondOrderAllPassFrequencyTransform._check(in_order, out_order, alpha, theta)
        theta *= torch.pi
        k = torch.arange(n_fft, device=device, dtype=torch.double)
        omega = k * (2 * torch.pi / n_fft)
        ww = SecondOrderAllPassFrequencyTransform.warp(omega, alpha, theta)
        dw = SecondOrderAllPassFrequencyTransform.diff_warp(omega, alpha, theta)

        m2 = k[: out_order + 1]
        wwm2 = ww.reshape(-1, 1) * m2.reshape(1, -1)
        real = torch.cos(wwm2) * dw.reshape(-1, 1)
        imag = -torch.sin(wwm2) * dw.reshape(-1, 1)

        A = torch.fft.ifft(torch.complex(real, imag), dim=0).real
        L = in_order + 1
        if 2 <= L:
            A[1:L] += A[-(L - 1) :].flip(0)
        A = A[:L]
        A[1:, 0] /= 2
        A[0, 1:] *= 2
        return None, None, (to(A, dtype=dtype),)

    @staticmethod
    def _forward(c, A):
        return torch.matmul(c, A)

    @staticmethod
    def warp(omega, alpha, theta):
        x = omega - theta
        y = omega + theta
        return (
            omega
            + torch.atan2(alpha * torch.sin(x), 1 - alpha * torch.cos(x))
            + torch.atan2(alpha * torch.sin(y), 1 - alpha * torch.cos(y))
        )

    @staticmethod
    def diff_warp(omega, alpha, theta):
        x = omega - theta
        y = omega + theta
        a1 = alpha
        a2 = alpha + alpha
        aa = alpha * alpha
        return (
            1
            + (a1 * torch.cos(x) - aa) / (1 - a2 * torch.cos(x) + aa)
            + (a1 * torch.cos(y) - aa) / (1 - a2 * torch.cos(y) + aa)
        )
