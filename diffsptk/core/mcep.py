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
from ..misc.utils import hankel
from ..misc.utils import is_power_of_two
from ..misc.utils import numpy_to_torch
from ..misc.utils import symmetric_toeplitz
from .freqt import FrequencyTransform


class CoefficientsFrequencyTransform(nn.Module):
    def __init__(self, in_order, out_order, alpha):
        super(CoefficientsFrequencyTransform, self).__init__()

        L1 = in_order + 1
        L2 = out_order + 1

        # Make transform matrix.
        A = np.zeros((L2, L1))
        A[:, 0] = (-alpha) ** np.arange(L2)
        for i in range(1, L2):
            i1 = i - 1
            for j in range(1, L1):
                j1 = j - 1
                A[i, j] = A[i1, j1] + alpha * (A[i, j1] - A[i1, j])

        self.register_buffer("A", numpy_to_torch(A.T))

    def forward(self, x):
        y = torch.matmul(x, self.A)
        return y


class MelCepstralAnalysis(nn.Module):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/mgcep.html>`_
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

    n_iter : int >= 0 [scalar]
        Number of iterations.

    """

    def __init__(self, cep_order, fft_length, alpha=0, n_iter=0):
        super(MelCepstralAnalysis, self).__init__()

        self.cep_order = cep_order
        self.fft_length = fft_length
        self.n_iter = n_iter

        assert 0 <= self.cep_order
        assert self.cep_order <= self.fft_length // 2
        assert is_power_of_two(self.fft_length)
        assert 0 <= self.n_iter

        self.freqt = FrequencyTransform(self.fft_length // 2, self.cep_order, alpha)
        self.ifreqt = FrequencyTransform(self.cep_order, self.fft_length // 2, -alpha)
        self.rfreqt = CoefficientsFrequencyTransform(
            self.fft_length // 2, 2 * self.cep_order, alpha
        )

        alpha_vector = (-alpha) ** np.arange(self.cep_order + 1)
        self.register_buffer("alpha_vector", numpy_to_torch(alpha_vector))

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
        >>> mcep = diffsptk.MelCepstralAnalysis(3, 16, 0.1, n_iter=1)
        >>> mc = mcep(stft(x))
        >>> mc
        tensor([[-0.8851,  0.7917, -0.1737,  0.0175],
                [-0.3522,  4.4222, -1.0882, -0.0511]])

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
