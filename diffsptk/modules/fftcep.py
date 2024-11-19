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
import torch.nn.functional as F

from ..misc.utils import check_size


class CepstralAnalysis(nn.Module):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/fftcep.html>`_
    for details.

    Parameters
    ----------
    cep_order : int >= 0
        Order of cepstrum, :math:`M`.

    fft_length : int >= 2M
        Number of FFT bins, :math:`L`.

    n_iter : int >= 0
        Number of iterations.

    accel : float >= 0
        Acceleration factor.

    References
    ----------
    .. [1] S. Imai et al., "Spectral envelope extraction by improved cepstral method,"
           *IEICE trans*, vol. J62-A, no. 4, pp. 217-223, 1979 (in Japanese).

    """

    def __init__(self, cep_order, fft_length, n_iter=0, accel=0):
        super().__init__()

        assert 0 <= cep_order <= fft_length // 2
        assert 0 <= n_iter
        assert 0 <= accel

        self.cep_order = cep_order
        self.fft_length = fft_length
        self.n_iter = n_iter
        self.accel = accel

    def forward(self, x):
        """Estimate cepstrum from spectrum.

        Parameters
        ----------
        x : Tensor [shape=(..., L/2+1)]
            Power spectrum.

        Returns
        -------
        out : Tensor [shape=(..., M+1)]
            Cepstrum.

        Examples
        --------
        >>> x = diffsptk.ramp(19)
        >>> stft = diffsptk.STFT(frame_length=10, frame_period=10, fft_length=16)
        >>> fftcep = diffsptk.CepstralAnalysis(3, 16)
        >>> c = fftcep(stft(x))
        >>> c
        tensor([[-0.9663,  0.8190, -0.0932, -0.0152],
                [-0.8539,  4.6173, -0.5496, -0.3207]])

        """
        check_size(x.size(-1), self.fft_length // 2 + 1, "dimension of spectrum")
        return self._forward(x, self.cep_order, self.n_iter, self.accel)

    @staticmethod
    def _forward(x, cep_order, n_iter, accel):
        N = cep_order + 1
        H = x.size(-1)

        e = torch.fft.irfft(torch.log(x))
        v = e[..., :N]
        e = F.pad(e[..., N:H], (N, 0))

        for _ in range(n_iter):
            e = torch.fft.hfft(e)
            e.masked_fill_(e < 0, 0)
            e = torch.fft.ihfft(e).real
            t = e[..., :N] * (1 + accel)
            v += t
            e -= F.pad(t, (0, H - N))

        indices = [0, N - 1] if H == N else [0]
        v[..., indices] *= 0.5
        return v

    _func = _forward
