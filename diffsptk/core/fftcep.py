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
import torch.nn as nn
import torch.nn.functional as F

from ..misc.utils import is_power_of_two


class CepstralAnalysis(nn.Module):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/fftcep.html>`_
    for details.

    Parameters
    ----------
    cep_order : int >= 0 [scalar]
        Order of cepstrum, :math:`M`.

    fft_length : int >= 2M [scalar]
        Number of FFT bins, :math:`L`.

    n_iter : int >= 0 [scalar]
        Number of iterations.

    accel : float >= 0 [scalar]
        Acceleration factor.

    """

    def __init__(self, cep_order, fft_length, n_iter=0, accel=0):
        super(CepstralAnalysis, self).__init__()

        self.cep_order = cep_order
        self.fft_length = fft_length
        self.n_iter = n_iter
        self.accel = 1 + accel

        assert 0 <= self.cep_order
        assert self.cep_order <= self.fft_length // 2
        assert is_power_of_two(self.fft_length)
        assert 0 <= self.n_iter
        assert 1 <= self.accel

    def forward(self, x):
        """Estimate cepstrum from spectrum.

        Parameters
        ----------
        x : Tensor [shape=(..., L/2+1)]
            Power spectrum.

        Returns
        -------
        v : Tensor [shape=(..., M+1)]
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
        # Torch's pad only supports 3D, 4D, 5D padding with non-constant padding.
        d = x.dim()
        for _ in range(3 - d):
            x = x.unsqueeze(0)

        M = self.cep_order
        H = self.fft_length // 2

        e = torch.fft.irfft(torch.log(x))
        v = e[..., : M + 1]
        e = F.pad(e[..., M + 1 : H + 1], (M + 1, 0))

        for _ in range(self.n_iter):
            e = torch.fft.hfft(e)
            e.masked_fill_(e < 0, 0)
            e = torch.fft.ihfft(e).real

            t = e[..., : M + 1] * self.accel
            v = v + t
            e = e - F.pad(t, (0, H - M))

        indices = [0, M] if H == M else [0]
        for m in indices:
            v[..., m] = 0.5 * v[..., m].clone()

        # Revert shape.
        for _ in range(3 - d):
            v = v.squeeze(0)
        return v
