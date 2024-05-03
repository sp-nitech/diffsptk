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


class MelCepstrumToMLSADigitalFilterCoefficients(nn.Module):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/mc2b.html>`_
    for details.

    Parameters
    ----------
    cep_order : int >= 0
        Order of cepstrum, :math:`M`.

    alpha : float in (-1, 1)
        Frequency warping factor, :math:`\\alpha`.

    """

    def __init__(self, cep_order, alpha=0):
        super().__init__()

        assert 0 <= cep_order
        assert abs(alpha) < 1

        self.cep_order = cep_order
        self.alpha = alpha

        # Make transform matrix.
        a = 1
        A = torch.eye(self.cep_order + 1, dtype=torch.double)
        for m in range(1, len(A)):
            a *= -self.alpha
            A[:, m:].fill_diagonal_(a)
        self.register_buffer("A", to(A.T))

    def forward(self, mc):
        """Convert mel-cepstrum to MLSA filter coefficients.

        Parameters
        ----------
        mc : Tensor [shape=(..., M+1)]
            Mel-cepstral coefficients.

        Returns
        -------
        out : Tensor [shape=(..., M+1)]
            MLSA filter coefficients.

        Examples
        --------
        >>> mc = diffsptk.ramp(4)
        >>> mc2b = diffsptk.MelCepstrumToMLSADigitalFilterCoefficients(4, 0.3)
        >>> b = mc2b(mc)
        >>> b
        tensor([-0.1686,  0.5620,  1.4600,  1.8000,  4.0000])

        """
        check_size(mc.size(-1), self.cep_order + 1, "dimension of cepstrum")
        return self._forward(mc, self.A)

    @staticmethod
    def _forward(mc, A):
        return torch.matmul(mc, A)

    def _func(mc, alpha):
        M = mc.size(-1) - 1
        b = torch.zeros_like(mc)
        b[..., M] = mc[..., M]
        for m in reversed(range(M)):
            b[..., m] = mc[..., m] - alpha * b[..., m + 1]
        return b
