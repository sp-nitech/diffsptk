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

from ..misc.utils import check_size
from ..misc.utils import default_dtype
from ..misc.utils import hankel
from ..misc.utils import is_power_of_two
from ..misc.utils import symmetric_toeplitz


def warp(omega, alpha, theta):
    x = omega - theta
    y = omega + theta
    return (
        omega
        + np.arctan2(alpha * np.sin(x), 1 - alpha * np.cos(x))
        + np.arctan2(alpha * np.sin(y), 1 - alpha * np.cos(y))
    )


class SecondOrderAllPassFrequencyTransform(nn.Module):
    def __init__(self, in_order, out_order, fft_length, alpha, theta):
        super(SecondOrderAllPassFrequencyTransform, self).__init__()

        assert 0 <= in_order
        assert 0 <= out_order
        assert in_order < fft_length
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
        delta = 2 * np.pi / fft_length
        omega = np.arange(fft_length) * delta
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
        self.register_buffer("A", torch.from_numpy(A.astype(default_dtype())))

    def forward(self, c1):
        c2 = torch.matmul(c1, self.A)
        return c2


class SecondOrderAllPassInverseFrequencyTransform(nn.Module):
    def __init__(self, in_order, out_order, fft_length, alpha, theta):
        super(SecondOrderAllPassInverseFrequencyTransform, self).__init__()

        assert 0 <= in_order
        assert 0 <= out_order
        assert out_order < fft_length
        assert abs(alpha) < 1
        assert 0 <= theta <= 1

        theta *= np.pi
        delta = 2 * np.pi / fft_length
        omega = np.arange(fft_length) * delta
        ww = warp(omega, alpha, theta)

        m1 = np.arange(-in_order, in_order + 1)
        wwm1 = ww.reshape(-1, 1) * m1.reshape(1, -1)
        real = np.cos(wwm1)
        imag = -np.sin(wwm1)

        M2 = out_order + 1
        A = np.fft.ifft(real + 1j * imag, axis=0).real
        A[:M2, in_order + 1 :] += np.flip(A[:M2, :in_order], axis=1)
        A = A[:M2, in_order:]
        A[0, 1:] /= 2
        A[1:, 0] *= 2
        self.register_buffer("A", torch.from_numpy(A.astype(default_dtype())).t())

    def forward(self, c2):
        c1 = torch.matmul(c2, self.A)
        return c1


class CoefficientsFrequencyTransform(nn.Module):
    def __init__(self, in_order, out_order, fft_length, alpha, theta):
        super(CoefficientsFrequencyTransform, self).__init__()

        assert 0 <= in_order
        assert 0 <= out_order
        assert out_order < fft_length
        assert abs(alpha) < 1
        assert 0 <= theta <= 1

        theta *= np.pi
        delta = 2 * np.pi / fft_length
        omega = np.arange(fft_length) * delta
        ww = warp(omega, alpha, theta)

        m2 = np.arange(out_order + 1)
        wwm2 = ww.reshape(-1, 1) * m2.reshape(1, -1)
        real = np.cos(wwm2)
        imag = -np.sin(wwm2)

        M1 = in_order + 1
        A = np.fft.ifft(real + 1j * imag, axis=0).real
        if 2 <= M1:
            A[1:M1] += np.flip(A[-(M1 - 1) :], axis=0)
        A = A[:M1]
        self.register_buffer("A", torch.from_numpy(A.astype(default_dtype())))

    def forward(self, c1):
        c2 = torch.matmul(c1, self.A)
        return c2


class SecondOrderAllPassMelCepstralAnalysis(nn.Module):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/smcep.html>`_
    for details. Note that the current implementation does not use the efficient
    Toeplitz-plus-Hankel system solver.

    Parameters
    ----------
    cep_order : int >= 0 [scalar]
        Order of mel-cepstrum, :math:`M`.

    fft_length : int >= 2M [scalar]
        Number of FFT bins, :math:`L`.

    alpha : float [-1 < alpha < 1]
        Frequency warping factor, :math:`\\alpha`.

    theta : float [0 <= theta <= 1]
        Emphasis frequency, :math:`\\theta`.

    n_iter : int >= 0 [scalar]
        Number of iterations.

    accuracy_factor : int >= 1 [scalar]
        Accuracy factor multiplied by FFT length.

    """

    def __init__(
        self, cep_order, fft_length, alpha=0, theta=0, n_iter=0, accuracy_factor=4
    ):
        super(SecondOrderAllPassMelCepstralAnalysis, self).__init__()

        self.cep_order = cep_order
        self.fft_length = fft_length
        self.n_iter = n_iter

        assert 0 <= self.cep_order
        assert self.cep_order <= self.fft_length // 2
        assert is_power_of_two(self.fft_length)
        assert 0 <= self.n_iter

        self.freqt = SecondOrderAllPassFrequencyTransform(
            self.fft_length // 2,
            self.cep_order,
            self.fft_length * accuracy_factor,
            alpha,
            theta,
        )
        self.ifreqt = SecondOrderAllPassInverseFrequencyTransform(
            self.cep_order,
            self.fft_length // 2,
            self.fft_length * accuracy_factor,
            alpha,
            theta,
        )
        self.rfreqt = CoefficientsFrequencyTransform(
            self.fft_length // 2,
            2 * self.cep_order,
            self.fft_length * accuracy_factor,
            alpha,
            theta,
        )

        seed = np.ones(1, dtype=default_dtype())
        alpha_vector = CoefficientsFrequencyTransform(
            0,
            self.cep_order,
            self.fft_length * accuracy_factor,
            alpha,
            theta,
        )(torch.tensor(seed))
        self.register_buffer("alpha_vector", alpha_vector)

    def forward(self, x):
        """Estimate mel-cepstrum from spectrum.

        Parameters
        ----------
        x : Tensor [shape=(..., L/2+1)]
            Power spectrum.

        Returns
        -------
        mc : Tensor [shape=(..., M+1)]
            Mel-cepstrum.

        Examples
        --------
        >>> x = diffsptk.ramp(19)
        >>> stft = diffsptk.STFT(frame_length=10, frame_period=10, fft_length=16)
        >>> smcep = diffsptk.SecondOrderAllPassMelCepstralAnalysis(3, 16, 0.1, n_iter=1)
        >>> mc = smcep(stft(x))
        >>> mc
        tensor([[-0.8851,  0.7917, -0.1737,  0.0175],
                [-0.3523,  4.4223, -1.0883, -0.0510]])

        """
        M = self.cep_order
        H = self.fft_length // 2
        check_size(x.size(-1), H + 1, "dimension of spectrum")

        log_x = torch.log(x)
        c = torch.fft.irfft(log_x)
        c[..., 0] *= 0.5
        c[..., H] *= 0.5
        mc = self.freqt(c[..., : H + 1])

        for _ in range(self.n_iter):
            c = self.ifreqt(mc)
            d = torch.fft.rfft(c, n=self.fft_length).real
            d = torch.exp(log_x - d - d)

            rd = torch.fft.irfft(d)
            rt = self.rfreqt(rd[..., : H + 1])
            r = rt[..., : M + 1]
            ra = r - self.alpha_vector

            R = symmetric_toeplitz(r)
            Q = hankel(rt)
            gradient = torch.linalg.solve(R + Q, ra)
            mc = mc + gradient

        return mc
