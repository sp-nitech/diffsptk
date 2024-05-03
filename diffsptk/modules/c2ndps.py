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


class CepstrumToNegativeDerivativeOfPhaseSpectrum(nn.Module):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/c2ndps.html>`_
    for details.

    Parameters
    ----------
    cep_order : int >= 0
        Order of cepstrum, :math:`M`.

    fft_length : int >= 2
        Number of FFT bins, :math:`L`.

    """

    def __init__(self, cep_order, fft_length):
        super().__init__()

        assert 0 <= cep_order
        assert max(1, cep_order) <= fft_length // 2

        self.cep_order = cep_order
        self.fft_length = fft_length
        ramp = self._precompute(self.cep_order, self.fft_length)
        self.register_buffer("ramp", ramp)

    def forward(self, c):
        """Convert cepstrum to NDPS.

        Parameters
        ----------
        c : Tensor [shape=(..., M+1)]
            Cepstrum.

        Returns
        -------
        out : Tensor [shape=(..., L/2+1)]
            NDPS.

        Examples
        --------
        >>> c = diffsptk.ramp(4)
        >>> c2ndps = diffsptk.CepstrumToNegativeDerivativeOfPhaseSpectrum(4, 8)
        >>> n = c2ndps(c)
        >>> n
        tensor([ 30.0000, -21.6569,  12.0000, -10.3431,  10.0000])

        """
        check_size(c.size(-1), self.cep_order + 1, "dimension of cepstrum")
        return self._forward(c, self.fft_length, self.ramp)

    @staticmethod
    def _forward(c, fft_length, ramp):
        v = c * ramp
        n = torch.fft.hfft(v, n=fft_length)[..., : fft_length // 2 + 1]
        return n

    @staticmethod
    def _func(c, fft_length):
        ramp = CepstrumToNegativeDerivativeOfPhaseSpectrum._precompute(
            c.size(-1) - 1, fft_length, dtype=c.dtype, device=c.device
        )
        return CepstrumToNegativeDerivativeOfPhaseSpectrum._forward(c, fft_length, ramp)

    @staticmethod
    def _precompute(cep_order, fft_length, dtype=None, device=None):
        half_fft_length = fft_length // 2
        ramp = torch.arange(cep_order + 1, dtype=torch.double, device=device) * 0.5
        if cep_order == half_fft_length:
            ramp[-1] *= 2
        return to(ramp, dtype=dtype)
