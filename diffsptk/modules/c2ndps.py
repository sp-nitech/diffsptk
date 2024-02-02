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

from ..misc.utils import check_size
from ..misc.utils import to


class CepstrumToNegativeDerivativeOfPhaseSpectrum(nn.Module):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/c2ndps.html>`_
    for details.

    Parameters
    ----------
    cep_order : int >= 0 [scalar]
        Order of cepstrum, :math:`M`.

    fft_length : int >= 2 [scalar]
        Number of FFT bins, :math:`L`.

    """

    def __init__(self, cep_order, fft_length, stateful=True):
        super(CepstrumToNegativeDerivativeOfPhaseSpectrum, self).__init__()

        self.cep_order = cep_order
        self.fft_length = fft_length

        assert 0 <= self.cep_order
        assert max(1, self.cep_order) <= self.fft_length // 2

        if stateful:
            ramp = self._make_ramp(self.cep_order, self.fft_length // 2)
            self.register_buffer("ramp", ramp)

    def forward(self, c):
        """Convert cepstrum to NDPS.

        Parameters
        ----------
        c : Tensor [shape=(..., M+1)]
            Cepstrum.

        Returns
        -------
        Tensor [shape=(..., L/2+1)]
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
        return self._forward(c, self.fft_length, ramp=getattr(self, "ramp", None))

    @staticmethod
    def _forward(c, fft_length, **kwargs):
        if kwargs.get("ramp") is None:
            ramp = CepstrumToNegativeDerivativeOfPhaseSpectrum._make_ramp(
                c.size(-1) - 1, fft_length // 2, dtype=c.dtype, device=c.device
            )
        else:
            ramp = kwargs["ramp"]
        v = c * ramp
        n = torch.fft.hfft(v, n=fft_length)[..., : fft_length // 2 + 1]
        return n

    @staticmethod
    def _make_ramp(cep_order, half_fft_length, dtype=None, device=None):
        ramp = torch.arange(cep_order + 1, dtype=torch.double, device=device) * 0.5
        if cep_order == half_fft_length:
            ramp[-1] *= 2
        return to(ramp, dtype=dtype)
