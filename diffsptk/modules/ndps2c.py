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

from ..misc.utils import to
from .base import BaseFunctionalModule
from .c2ndps import CepstrumToNegativeDerivativeOfPhaseSpectrum


class NegativeDerivativeOfPhaseSpectrumToCepstrum(BaseFunctionalModule):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/ndps2c.html>`_
    for details.

    Parameters
    ----------
    cep_order : int >= 0
        The order of the cepstrum, :math:`M`.

    fft_length : int >= 2
        The number of FFT bins, :math:`L`.

    References
    ----------
    .. [1] B. Yegnanarayana, "Pole-zero decomposition of speech spectra," *Signal
           Processing*, vol. 3, no. 1, pp. 5-17, 1981.

    """

    def __init__(self, cep_order, fft_length):
        super().__init__()

        self.values, tensors = self._precompute(cep_order, fft_length)
        self.register_buffer("ramp", tensors[0])

    def forward(self, n):
        """Convert NPDS to cepstrum.

        Parameters
        ----------
        n : Tensor [shape=(..., L/2+1)]
            The NDPS.

        Returns
        -------
        out : Tensor [shape=(..., M+1)]
            The cepstrum.

        Examples
        --------
        >>> n = diffsptk.ramp(4)
        >>> ndps2c = diffsptk.NegativeDerivativeOfPhaseSpectrumToCepstrum(4, 8)
        >>> c = ndps2c(n)
        >>> c
        tensor([ 0.0000, -1.7071,  0.0000, -0.0976,  0.0000])

        """
        return self._forward(n, *self.values, **self._buffers)

    @staticmethod
    def _func(n, cep_order):
        values, tensors = NegativeDerivativeOfPhaseSpectrumToCepstrum._precompute(
            cep_order, 2 * (n.size(-1) - 1), dtype=n.dtype, device=n.device
        )
        return NegativeDerivativeOfPhaseSpectrumToCepstrum._forward(
            n, *values, *tensors
        )

    @staticmethod
    def _check(cep_order, fft_length):
        CepstrumToNegativeDerivativeOfPhaseSpectrum._check(cep_order, fft_length)

    @staticmethod
    def _precompute(cep_order, fft_length, dtype=None, device=None):
        NegativeDerivativeOfPhaseSpectrumToCepstrum._check(cep_order, fft_length)
        half_fft_length = fft_length // 2
        ramp = torch.arange(cep_order + 1, dtype=torch.double, device=device)
        ramp *= half_fft_length
        if cep_order == half_fft_length:
            ramp[-1] *= 2
        ramp[1:] = 1 / ramp[1:]
        return (cep_order,), (to(ramp, dtype=dtype),)

    @staticmethod
    def _forward(n, cep_order, ramp):
        c = torch.fft.hfft(n)[..., : cep_order + 1]
        c *= ramp
        return c
