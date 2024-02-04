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


class CepstrumToAutocorrelation(nn.Module):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/c2acr.html>`_
    for details.

    Parameters
    ----------
    cep_order : int >= 0
        Order of cepstrum, :math:`M`.

    acr_order : int >= 0
        Order of autocorrelation, :math:`N`.

    n_fft : int >> :math:`N`
        Number of FFT bins. Accurate conversion requires the large value.

    """

    def __init__(self, cep_order, acr_order, n_fft=512):
        super(CepstrumToAutocorrelation, self).__init__()

        self.cep_order = cep_order
        self.acr_order = acr_order
        self.n_fft = n_fft

        assert 0 <= self.cep_order
        assert 0 <= self.acr_order
        assert max(self.cep_order + 1, self.acr_order + 1) < self.n_fft

    def forward(self, c):
        """Convert cepstrum to autocorrelation.

        Parameters
        ----------
        c : Tensor [shape=(..., M+1)]
            Cepstral coefficients.

        Returns
        -------
        Tensor [shape=(..., N+1)]
            Autocorrelation.

        Examples
        --------
        >>> c = diffsptk.nrand(4)
        >>> c
        tensor([-0.1751,  0.1950, -0.3211,  0.3523, -0.5453])
        >>> c2acr = diffsptk.CepstrumToAutocorrelation(4, 16)
        >>> r = c2acr(c)
        >>> r
        tensor([ 1.0672, -0.0485, -0.1564,  0.2666, -0.4551])

        """
        check_size(c.size(-1), self.cep_order + 1, "dimension of cepstrum")
        return self._forward(c, self.acr_order, self.n_fft)

    @staticmethod
    def _forward(c, acr_order, n_fft):
        x = torch.fft.rfft(c, n=n_fft).real
        x = torch.exp(2 * x)
        r = torch.fft.hfft(x, norm="forward")[..., : acr_order + 1]
        return r
