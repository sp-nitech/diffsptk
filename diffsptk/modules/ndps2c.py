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

from ..misc.utils import check_size
from ..misc.utils import get_values
from ..misc.utils import to
from .base import BaseFunctionalModule


class NegativeDerivativeOfPhaseSpectrumToCepstrum(BaseFunctionalModule):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/ndps2c.html>`_
    for details.

    Parameters
    ----------
    fft_length : int >= 2
        The number of FFT bins, :math:`L`.

    cep_order : int >= 0
        The order of the cepstrum, :math:`M`.

    References
    ----------
    .. [1] B. Yegnanarayana, "Pole-zero decomposition of speech spectra," *Signal
           Processing*, vol. 3, no. 1, pp. 5-17, 1981.

    """

    def __init__(self, fft_length, cep_order):
        super().__init__()

        self.in_dim = fft_length // 2 + 1

        self.values, _, tensors = self._precompute(*get_values(locals()))
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
        check_size(n.size(-1), self.in_dim, "dimension of spectrum")
        return self._forward(n, *self.values, **self._buffers)

    @staticmethod
    def _func(n, *args, **kwargs):
        values, _, tensors = NegativeDerivativeOfPhaseSpectrumToCepstrum._precompute(
            2 * n.size(-1) - 2, *args, **kwargs, device=n.device, dtype=n.dtype
        )
        return NegativeDerivativeOfPhaseSpectrumToCepstrum._forward(
            n, *values, *tensors
        )

    @staticmethod
    def _takes_input_size():
        return True

    @staticmethod
    def _check(fft_length, cep_order):
        if fft_length // 2 < max(1, cep_order):
            raise ValueError(
                "half of fft_length must be greater than or equal to cep_order."
            )
        if cep_order < 0:
            raise ValueError("cep_order must be non-negative.")

    @staticmethod
    def _precompute(fft_length, cep_order, device=None, dtype=None):
        NegativeDerivativeOfPhaseSpectrumToCepstrum._check(fft_length, cep_order)
        half_fft_length = fft_length // 2
        ramp = torch.arange(cep_order + 1, device=device, dtype=torch.double)
        ramp *= half_fft_length
        if cep_order == half_fft_length:
            ramp[-1] *= 2
        ramp[1:] = 1 / ramp[1:]
        return (cep_order,), None, (to(ramp, dtype=dtype),)

    @staticmethod
    def _forward(n, cep_order, ramp):
        c = torch.fft.hfft(n)[..., : cep_order + 1]
        c *= ramp
        return c
