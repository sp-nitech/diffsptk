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

from ..misc.utils import numpy_to_torch


def warp(omega, alpha, theta):
    """Warp frequency.

    Parameters
    ----------
    omega : float [0 <= omega <= 2pi]
        Frequency.

    alpha : float [-1 < alpha < 1]
        Frequency warping factor, :math:`\\alpha`.

    theta : float [0 <= theta <= 1]
        Emphasis frequency, :math:`\\theta`.

    Returns
    -------
    w : float
        Warped frequency.

    """
    x = omega - theta
    y = omega + theta
    w = (
        omega
        + np.arctan2(alpha * np.sin(x), 1 - alpha * np.cos(x))
        + np.arctan2(alpha * np.sin(y), 1 - alpha * np.cos(y))
    )
    return w


class SecondOrderAllPassFrequencyTransform(nn.Module):
    """Second-order all-pass frequecy transform module.

    Parameters
    ----------
    in_order : int >= 0 [scalar]
        Order of input sequence, :math:`M_1`.

    out_order : int >= 0 [scalar]
        Order of output sequence, :math:`M_2`.

    alpha : float [-1 < alpha < 1]
        Frequency warping factor, :math:`\\alpha`.

    theta : float [0 <= theta <= 1]
        Emphasis frequency, :math:`\\theta`.

    n_fft : int >> :math:`M_2` [scalar]
        Number of FFT bins. Accurate conversion requires the large value.

    """

    def __init__(self, in_order, out_order, alpha=0, theta=0, n_fft=512):
        super(SecondOrderAllPassFrequencyTransform, self).__init__()

        assert 0 <= in_order
        assert 0 <= out_order
        assert out_order < n_fft
        assert abs(alpha) < 1
        assert 0 <= theta <= 1

        def diff_warp(omega, alpha, theta):
            x = omega - theta
            y = omega + theta
            a1 = alpha
            a2 = alpha + alpha
            aa = alpha * alpha
            return (
                1
                + (a1 * np.cos(x) - aa) / (1 - a2 * np.cos(x) + aa)
                + (a1 * np.cos(y) - aa) / (1 - a2 * np.cos(y) + aa)
            )

        theta *= np.pi
        delta = 2 * np.pi / n_fft
        omega = np.arange(n_fft) * delta
        ww = warp(omega, alpha, theta)
        dw = diff_warp(omega, alpha, theta)

        m2 = np.arange(out_order + 1)
        wwm2 = ww.reshape(-1, 1) * m2.reshape(1, -1)
        real = np.cos(wwm2) * dw.reshape(-1, 1)
        imag = -np.sin(wwm2) * dw.reshape(-1, 1)

        M1 = in_order + 1
        A = np.fft.ifft(real + 1j * imag, axis=0).real
        if 2 <= M1:
            A[1:M1] += np.flip(A[-(M1 - 1) :], axis=0)
        A = A[:M1]
        A[1:, 0] /= 2
        A[0, 1:] *= 2
        self.register_buffer("A", numpy_to_torch(A))

    def forward(self, c1):
        """Perform second-order all-pass frequency transform.

        Parameters
        ----------
        c1 : Tensor [shape=(..., M1+1)]
            Input sequence.

        Returns
        -------
        c2 : Tensor [shape=(..., M2+1)]
            Warped sequence.

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
        c2 = torch.matmul(c1, self.A)
        return c2
