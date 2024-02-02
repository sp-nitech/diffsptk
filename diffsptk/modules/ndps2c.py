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

    def __init__(self, cep_order, fft_length, stateful=True):
        super(NegativeDerivativeOfPhaseSpectrumToCepstrum, self).__init__()

        self.cep_order = cep_order
        self.half_fft_length = fft_length // 2

        assert 0 <= self.cep_order
        assert max(1, self.cep_order) <= self.half_fft_length

        if stateful:
            ramp = self._make_ramp(self.cep_order, self.half_fft_length)
            self.register_buffer("ramp", ramp)

    def forward(self, n):
        """Convert NPDS to cepstrum.

        Parameters
        ----------
        n : Tensor [shape=(..., L/2+1)]
            NDPS.

        Returns
        -------
        Tensor [shape=(..., M+1)]
            Cepstrum.

        Examples
        --------
        >>> n = diffsptk.ramp(4)
        >>> ndps2c = diffsptk.NegativeDerivativeOfPhaseSpectrumToCepstrum(4, 8)
        >>> c = ndps2c(n)
        >>> c
        tensor([ 0.0000, -1.7071,  0.0000, -0.0976,  0.0000])

        """
        check_size(n.size(-1), self.half_fft_length + 1, "dimension of spectrum")
        return self._forward(n, self.cep_order, ramp=getattr(self, "ramp", None))

    @staticmethod
    def _forward(n, cep_order, **kwargs):
        c = torch.fft.hfft(n)[..., : cep_order + 1]
        if kwargs.get("ramp") is None:
            ramp = NegativeDerivativeOfPhaseSpectrumToCepstrum._make_ramp(
                cep_order, n.size(-1) - 1, dtype=n.dtype, device=n.device
            )
        else:
            ramp = kwargs["ramp"]
        c *= ramp
        return c

    @staticmethod
    def _make_ramp(cep_order, half_fft_length, dtype=None, device=None):
        ramp = torch.arange(cep_order + 1, dtype=torch.double, device=device)
        ramp *= half_fft_length
        if cep_order == half_fft_length:
            ramp[-1] *= 2
        ramp[1:] = 1 / ramp[1:]
        return to(ramp, dtype=dtype)
