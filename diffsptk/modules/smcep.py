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
from ..misc.utils import hankel
from ..misc.utils import symmetric_toeplitz
from ..misc.utils import to
from .freqt2 import SecondOrderAllPassFrequencyTransform
from .ifreqt2 import SecondOrderAllPassInverseFrequencyTransform


class CoefficientsFrequencyTransform(nn.Module):
    def __init__(self, in_order, out_order, alpha, theta, n_fft=512):
        super().__init__()

        theta *= torch.pi

        k = torch.arange(n_fft, dtype=torch.double)
        omega = k * (2 * torch.pi / n_fft)
        ww = SecondOrderAllPassFrequencyTransform.warp(omega, alpha, theta)

        m2 = k[: out_order + 1]
        wwm2 = ww.reshape(-1, 1) * m2.reshape(1, -1)
        real = torch.cos(wwm2)
        imag = -torch.sin(wwm2)

        A = torch.fft.ifft(torch.complex(real, imag), dim=0).real
        L = in_order + 1
        if 2 <= L:
            A[1:L] += A[-(L - 1) :].flip(0)
        A = A[:L]
        self.register_buffer("A", to(A))

    def forward(self, c):
        return torch.matmul(c, self.A)


class SecondOrderAllPassMelCepstralAnalysis(nn.Module):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/smcep.html>`_
    for details. Note that the current implementation does not use the efficient
    Toeplitz-plus-Hankel system solver.

    Parameters
    ----------
    cep_order : int >= 0
        Order of mel-cepstrum, :math:`M`.

    fft_length : int >= 2M
        Number of FFT bins, :math:`L`.

    alpha : float in (-1, 1)
        Frequency warping factor, :math:`\\alpha`.

    theta : float in [0, 1]
        Emphasis frequency, :math:`\\theta`.

    n_iter : int >= 0
        Number of iterations.

    accuracy_factor : int >= 1
        Accuracy factor multiplied by FFT length.

    References
    ----------
    .. [1] T. Wakako et al., "Speech spectral estimation based on expansion of log
           spectrum by arbitrary basis functions," *IEICE Trans*, vol. J82-D-II, no. 12,
           pp. 2203-2211, 1999 (in Japanese).

    """

    def __init__(
        self, cep_order, fft_length, alpha=0, theta=0, n_iter=0, accuracy_factor=4
    ):
        super().__init__()

        assert 0 <= cep_order <= fft_length // 2
        assert 0 <= n_iter

        self.cep_order = cep_order
        self.fft_length = fft_length
        self.n_iter = n_iter

        n_fft = fft_length * accuracy_factor

        self.freqt = SecondOrderAllPassFrequencyTransform(
            self.fft_length // 2,
            self.cep_order,
            alpha,
            theta,
            n_fft=n_fft,
        )
        self.ifreqt = SecondOrderAllPassInverseFrequencyTransform(
            self.cep_order,
            self.fft_length // 2,
            alpha,
            theta,
            n_fft=n_fft,
        )
        self.rfreqt = CoefficientsFrequencyTransform(
            self.fft_length // 2,
            2 * self.cep_order,
            alpha,
            theta,
            n_fft=n_fft,
        )

        seed = torch.ones(1)
        alpha_vector = CoefficientsFrequencyTransform(
            0,
            self.cep_order,
            alpha,
            theta,
            n_fft=n_fft,
        )(seed)
        self.register_buffer("alpha_vector", alpha_vector)

    def forward(self, x):
        """Estimate mel-cepstrum from spectrum.

        Parameters
        ----------
        x : Tensor [shape=(..., L/2+1)]
            Power spectrum.

        Returns
        -------
        out : Tensor [shape=(..., M+1)]
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
