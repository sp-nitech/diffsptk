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

from ..misc.utils import check_size
from ..misc.utils import get_values
from ..misc.utils import to
from .base import BaseFunctionalModule


class MLSADigitalFilterCoefficientsToMelCepstrum(BaseFunctionalModule):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/b2mc.html>`_
    for details.

    Parameters
    ----------
    cep_order : int >= 0
        The order of the cepstrum, :math:`M`.

    alpha : float in (-1, 1)
        The frequency warping factor, :math:`\\alpha`.

    References
    ----------
    .. [1] K. Tokuda et al., "Spectral estimation of speech by mel-generalized cepstral
           analysis," *Electronics and Communications in Japan, part 3*, vol. 76, no. 2,
           pp. 30-43, 1993.

    """

    def __init__(self, cep_order, alpha=0, device=None, dtype=None):
        super().__init__()

        self.in_dim = cep_order + 1

        _, _, tensors = self._precompute(*get_values(locals()))
        self.register_buffer("A", tensors[0])

    def forward(self, b):
        """Convert MLSA filter coefficients to mel-cepstrum.

        Parameters
        ----------
        b : Tensor [shape=(..., M+1)]
            The MLSA filter coefficients.

        Returns
        -------
        out : Tensor [shape=(..., M+1)]
            The mel-cepstral coefficients.

        Examples
        --------
        >>> b = diffsptk.ramp(4)
        >>> mc2b = diffsptk.MelCepstrumToMLSADigitalFilterCoefficients(4, 0.3)
        >>> b2mc = diffsptk.MLSADigitalFilterCoefficientsToMelCepstrum(4, 0.3)
        >>> b2 = mc2b(b2mc(b))
        >>> b2
        tensor([0.0000, 1.0000, 2.0000, 3.0000, 4.0000])

        """
        check_size(b.size(-1), self.in_dim, "dimension of cepstrum")
        return self._forward(b, **self._buffers)

    @staticmethod
    def _func(b, alpha):
        MLSADigitalFilterCoefficientsToMelCepstrum._check(b.size(-1) - 1, alpha)
        return b + F.pad(alpha * b[..., 1:], (0, 1))

    @staticmethod
    def _check(cep_order, alpha):
        if cep_order < 0:
            raise ValueError("cep_order must be non-negative.")
        if 1 <= abs(alpha):
            raise ValueError("alpha must be in (-1, 1).")

    @staticmethod
    def _precompute(cep_order, alpha, device=None, dtype=None):
        MLSADigitalFilterCoefficientsToMelCepstrum._check(cep_order, alpha)
        A = torch.eye(cep_order + 1, device=device, dtype=torch.double)
        A[:, 1:].fill_diagonal_(alpha)
        return None, None, (to(A.T, dtype=dtype),)

    @staticmethod
    def _forward(b, A):
        return torch.matmul(b, A)
