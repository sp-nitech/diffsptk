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

from ..typing import Precomputed
from ..utils.private import check_size, filter_values, to
from .b2mc import MLSADigitalFilterCoefficientsToMelCepstrum
from .base import BaseFunctionalModule


class MelCepstrumToMLSADigitalFilterCoefficients(BaseFunctionalModule):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/mc2b.html>`_
    for details.

    Parameters
    ----------
    cep_order : int >= 0
        The order of the cepstrum, :math:`M`.

    alpha : float in (-1, 1)
        The frequency warping factor, :math:`\\alpha`.

    device : torch.device or None
        The device of this module.

    dtype : torch.dtype or None
        The data type of this module.

    References
    ----------
    .. [1] K. Tokuda et al., "Spectral estimation of speech by mel-generalized cepstral
           analysis," *Electronics and Communications in Japan, part 3*, vol. 76, no. 2,
           pp. 30-43, 1993.

    """

    def __init__(
        self,
        cep_order: int,
        alpha: float = 0,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        self.in_dim = cep_order + 1

        _, _, tensors = self._precompute(**filter_values(locals()))
        self.register_buffer("A", tensors[0])

    def forward(self, mc: torch.Tensor) -> torch.Tensor:
        """Convert mel-cepstrum to MLSA filter coefficients.

        Parameters
        ----------
        mc : Tensor [shape=(..., M+1)]
            The mel-cepstral coefficients.

        Returns
        -------
        out : Tensor [shape=(..., M+1)]
            The MLSA filter coefficients.

        Examples
        --------
        >>> mc = diffsptk.ramp(4)
        >>> mc2b = diffsptk.MelCepstrumToMLSADigitalFilterCoefficients(4, 0.3)
        >>> b = mc2b(mc)
        >>> b
        tensor([-0.1686,  0.5620,  1.4600,  1.8000,  4.0000])

        """
        check_size(mc.size(-1), self.in_dim, "dimension of cepstrum")
        return self._forward(mc, **self._buffers)

    @staticmethod
    def _func(mc: torch.Tensor, alpha: float) -> torch.Tensor:
        M = mc.size(-1) - 1
        MelCepstrumToMLSADigitalFilterCoefficients._check(M, alpha)
        b = torch.zeros_like(mc)
        b[..., M] = mc[..., M]
        for m in reversed(range(M)):
            b[..., m] = mc[..., m] - alpha * b[..., m + 1]
        return b

    @staticmethod
    def _takes_input_size() -> bool:
        return True

    @staticmethod
    def _check(*args, **kwargs) -> None:
        MLSADigitalFilterCoefficientsToMelCepstrum._check(*args, **kwargs)

    @staticmethod
    def _precompute(
        cep_order: int,
        alpha: float,
        device: torch.device | None,
        dtype: torch.dtype | None,
    ) -> Precomputed:
        MelCepstrumToMLSADigitalFilterCoefficients._check(cep_order, alpha)
        a = 1
        A = torch.eye(cep_order + 1, device=device, dtype=torch.double)
        for m in range(1, len(A)):
            a *= -alpha
            A[:, m:].fill_diagonal_(a)
        return None, None, (to(A.T, dtype=dtype),)

    @staticmethod
    def _forward(mc: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        return torch.matmul(mc, A)
