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
from .b2mc import MLSADigitalFilterCoefficientsToMelCepstrum
from .gnorm import GeneralizedCepstrumGainNormalization
from .ignorm import GeneralizedCepstrumInverseGainNormalization
from .mc2b import MelCepstrumToMLSADigitalFilterCoefficients
from .mcep import MelCepstralAnalysis
from .mgc2mgc import MelGeneralizedCepstrumToMelGeneralizedCepstrum


class CoefficientsFrequencyTransform(nn.Module):
    def __init__(self, in_order, out_order, alpha):
        super().__init__()

        beta = 1 - alpha * alpha
        L1 = in_order + 1
        L2 = out_order + 1

        # Make transform matrix.
        A = torch.zeros((L2, L1), dtype=torch.double)
        A[0, 0] = 1
        if 1 < L2 and 1 < L1:
            A[1, 1:] = alpha ** torch.arange(L1 - 1, dtype=torch.double) * beta
        for i in range(2, L2):
            i1 = i - 1
            for j in range(1, L1):
                j1 = j - 1
                A[i, j] = A[i1, j1] + alpha * (A[i, j1] - A[i1, j])

        self.register_buffer("A", to(A.T))

    def forward(self, x):
        y = torch.matmul(x, self.A)
        return y


class PTransform(nn.Module):
    def __init__(self, order, alpha):
        super().__init__()

        # Make transform matrix.
        A = torch.eye(order + 1, dtype=torch.double)
        A[:, 1:].fill_diagonal_(alpha)

        A[0, 0] -= alpha * alpha
        A[0, 1] += alpha
        A[-1, -1] += alpha

        self.register_buffer("A", to(A.T))

    def forward(self, p):
        p = torch.matmul(p, self.A)
        return p


class QTransform(nn.Module):
    def __init__(self, order, alpha):
        super().__init__()

        # Make transform matrix.
        A = torch.eye(order + 1, dtype=torch.double)
        A[1:].fill_diagonal_(alpha)

        A[1, 0] = 0
        A[1, 1] += alpha

        self.register_buffer("A", to(A.T))

    def forward(self, q):
        q = torch.matmul(q, self.A)
        return q


class MelGeneralizedCepstralAnalysis(nn.Module):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/mgcep.html>`_
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

    gamma : float in [-1, 0]
        Gamma, :math:`\\gamma`.

    n_iter : int >= 0
        Number of iterations.

    """

    def __init__(self, cep_order, fft_length, alpha=0, gamma=0, n_iter=0):
        super().__init__()

        assert 0 <= cep_order <= fft_length // 2
        assert gamma <= 0
        assert 0 <= n_iter

        self.cep_order = cep_order
        self.fft_length = fft_length
        self.gamma = gamma
        self.n_iter = n_iter

        if gamma == 0:
            self.mcep = MelCepstralAnalysis(cep_order, fft_length, alpha, n_iter=n_iter)
        else:
            self.cfreqt = CoefficientsFrequencyTransform(
                cep_order, fft_length - 1, -alpha
            )
            self.pfreqt = CoefficientsFrequencyTransform(
                fft_length - 1, 2 * cep_order, alpha
            )
            self.rfreqt = CoefficientsFrequencyTransform(
                fft_length - 1, cep_order, alpha
            )

            self.ptrans = PTransform(2 * cep_order, alpha)
            self.qtrans = QTransform(2 * cep_order, alpha)

            self.b2b = nn.Sequential(
                GeneralizedCepstrumInverseGainNormalization(cep_order, -1),
                MLSADigitalFilterCoefficientsToMelCepstrum(cep_order, alpha),
                MelGeneralizedCepstrumToMelGeneralizedCepstrum(
                    cep_order, cep_order, in_gamma=-1, out_gamma=gamma
                ),
                MelCepstrumToMLSADigitalFilterCoefficients(cep_order, alpha),
                GeneralizedCepstrumGainNormalization(cep_order, gamma),
            )

            self.b2mc = nn.Sequential(
                GeneralizedCepstrumInverseGainNormalization(cep_order, gamma),
                MLSADigitalFilterCoefficientsToMelCepstrum(cep_order, alpha),
            )

    def forward(self, x):
        """Estimate mel-generalized cepstrum from spectrum.

        Parameters
        ----------
        x : Tensor [shape=(..., L/2+1)]
            Power spectrum.

        Returns
        -------
        out : Tensor [shape=(..., M+1)]
            Mel-generalized cepstrum.

        Examples
        --------
        >>> x = diffsptk.ramp(19)
        >>> stft = diffsptk.STFT(frame_length=10, frame_period=10, fft_length=16)
        >>> mgcep = diffsptk.MelGeneralizedCepstralAnalysis(3, 16, 0.1, n_iter=1)
        >>> mc = mgcep(stft(x))
        >>> mc
        tensor([[-0.8851,  0.7917, -0.1737,  0.0175],
                [-0.3522,  4.4222, -1.0882, -0.0511]])

        """
        if self.gamma == 0:
            mc = self.mcep(x)
            return mc

        M = self.cep_order
        H = self.fft_length // 2
        check_size(x.size(-1), H + 1, "dimension of spectrum")

        def newton(gamma, b1):
            def epsilon(gamma, r, b):
                eps = r[..., 0] + gamma * (r[..., 1:] * b).sum(-1)
                return eps

            b0 = torch.zeros(*b1.shape[:-1], 1, device=b1.device)
            b = torch.cat((b0, b1), dim=-1)
            c = self.cfreqt(b)
            C = torch.fft.rfft(c, n=self.fft_length)

            if gamma == -1:
                p_re = x
            else:
                X = 1 + gamma * C.real
                Y = gamma * C.imag
                XX = X * X
                YY = Y * Y
                D = XX + YY
                E = torch.pow(D, -1 / gamma)

                p = x * E / D
                p_re = p
                q = p / D
                q_re = q * (XX - YY)
                q_im = q * (2 * X * Y)
                r_re = p * X
                r_im = p * Y

            p = self.pfreqt(torch.fft.irfft(p_re))
            if gamma == -1:
                q = p
                r = p[..., : M + 1]
            else:
                q = self.pfreqt(torch.fft.irfft(torch.complex(q_re, q_im)))
                r = self.rfreqt(torch.fft.irfft(torch.complex(r_re, r_im)))

            p = self.ptrans(p)
            q = self.qtrans(q)

            if gamma != -1:
                eps = epsilon(gamma, r, b1)

            pt = p[..., :M]
            qt = q[..., 2:] * (1 + gamma)
            rt = r[..., 1:]

            R = symmetric_toeplitz(pt)
            Q = hankel(qt)
            gradient = torch.linalg.solve(R + Q, rt)
            b1 = b1 + gradient

            if gamma == -1:
                eps = epsilon(gamma, r, b1)

            b0 = torch.sqrt(eps).unsqueeze(-1)
            return b0, b1

        b1 = torch.zeros(*x.shape[:-1], M, device=x.device)
        b0, b1 = newton(-1, b1)

        if self.gamma != -1:
            b = torch.cat((b0, b1), dim=-1)
            b = self.b2b(b)
            _, b1 = torch.split(b, [1, M], dim=-1)
            for _ in range(self.n_iter):
                b0, b1 = newton(self.gamma, b1)

        b = torch.cat((b0, b1), dim=-1)
        mc = self.b2mc(b)
        return mc
