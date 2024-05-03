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


class NegativeDerivativeOfPhaseSpectrumToCepstrum(nn.Module):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/ndps2c.html>`_
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

    def forward(self, n):
        """Convert NPDS to cepstrum.

        Parameters
        ----------
        n : Tensor [shape=(..., L/2+1)]
            NDPS.

        Returns
        -------
        out : Tensor [shape=(..., M+1)]
            Cepstrum.

        Examples
        --------
        >>> n = diffsptk.ramp(4)
        >>> ndps2c = diffsptk.NegativeDerivativeOfPhaseSpectrumToCepstrum(4, 8)
        >>> c = ndps2c(n)
        >>> c
        tensor([ 0.0000, -1.7071,  0.0000, -0.0976,  0.0000])

        """
        check_size(n.size(-1), self.fft_length // 2 + 1, "dimension of spectrum")
        return self._forward(n, self.cep_order, self.ramp)

    @staticmethod
    def _forward(n, cep_order, ramp):
        c = torch.fft.hfft(n)[..., : cep_order + 1]
        c *= ramp
        return c

    @staticmethod
    def _func(n, cep_order):
        ramp = NegativeDerivativeOfPhaseSpectrumToCepstrum._precompute(
            cep_order, 2 * (n.size(-1) - 1), dtype=n.dtype, device=n.device
        )
        return NegativeDerivativeOfPhaseSpectrumToCepstrum._forward(n, cep_order, ramp)

    @staticmethod
    def _precompute(cep_order, fft_length, dtype=None, device=None):
        half_fft_length = fft_length // 2
        ramp = torch.arange(cep_order + 1, dtype=torch.double, device=device)
        ramp *= half_fft_length
        if cep_order == half_fft_length:
            ramp[-1] *= 2
        ramp[1:] = 1 / ramp[1:]
        return to(ramp, dtype=dtype)
