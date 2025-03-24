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

from ..typing import Precomputed
from ..utils.private import check_size, get_values
from .base import BaseFunctionalModule


class CepstralAnalysis(BaseFunctionalModule):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/fftcep.html>`_
    for details.

    Parameters
    ----------
    fft_length : int >= 2M
        The number of FFT bins, :math:`L`.

    cep_order : int >= 0
        The order of the cepstrum, :math:`M`.

    accel : float >= 0
        The acceleration factor.

    n_iter : int >= 0
        The number of iterations.

    References
    ----------
    .. [1] S. Imai et al., "Spectral envelope extraction by improved cepstral method,"
           *IEICE trans*, vol. J62-A, no. 4, pp. 217-223, 1979 (in Japanese).

    """

    def __init__(
        self, *, fft_length: int, cep_order: int, accel: float = 0, n_iter: int = 0
    ) -> None:
        super().__init__()

        self.in_dim = fft_length // 2 + 1

        self.values = self._precompute(*get_values(locals()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform cepstral analysis.

        Parameters
        ----------
        x : Tensor [shape=(..., L/2+1)]
            The power spectrum.

        Returns
        -------
        out : Tensor [shape=(..., M+1)]
            The cepstrum.

        Examples
        --------
        >>> x = diffsptk.ramp(19)
        >>> stft = diffsptk.STFT(frame_length=10, frame_period=10, fft_length=16)
        >>> fftcep = diffsptk.CepstralAnalysis(fft_length=16, cep_order=3)
        >>> c = fftcep(stft(x))
        >>> c
        tensor([[-0.9663,  0.8190, -0.0932, -0.0152],
                [-0.8539,  4.6173, -0.5496, -0.3207]])

        """
        check_size(x.size(-1), self.in_dim, "dimension of spectrum")
        return self._forward(x, *self.values)

    @staticmethod
    def _func(x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        values = CepstralAnalysis._precompute(2 * x.size(-1) - 2, *args, **kwargs)
        return CepstralAnalysis._forward(x, *values)

    @staticmethod
    def _takes_input_size() -> bool:
        return True

    @staticmethod
    def _check(fft_length: int, cep_order: int, accel: float, n_iter: int) -> None:
        if fft_length <= 1:
            raise ValueError("fft_length must be greater than 1.")
        if cep_order < 0:
            raise ValueError("cep_order must be non-negative.")
        if fft_length < 2 * cep_order:
            raise ValueError("cep_order must be less than or equal to fft_length // 2.")
        if accel < 0:
            raise ValueError("accel must be non-negative.")
        if n_iter < 0:
            raise ValueError("n_iter must be non-negative.")

    @staticmethod
    def _precompute(
        fft_length: int, cep_order: int, accel: float, n_iter: int
    ) -> Precomputed:
        CepstralAnalysis._check(fft_length, cep_order, accel, n_iter)
        return (cep_order, accel, n_iter)

    @staticmethod
    def _forward(
        x: torch.Tensor, cep_order: int, accel: float, n_iter: int
    ) -> torch.Tensor:
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
