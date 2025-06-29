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
from ..utils.private import check_size, filter_values, get_gamma
from .base import BaseFunctionalModule


class GeneralizedCepstrumGainNormalization(BaseFunctionalModule):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/gnorm.html>`_
    for details.

    Parameters
    ----------
    cep_order : int >= 0
        The order of the cepstrum, :math:`M`.

    gamma : float in [-1, 1]
        The gamma parameter, :math:`\\gamma`.

    c : int >= 1 or None
        The number of filter stages.

    References
    ----------
    .. [1] T. Kobayashi et al., "Spectral analysis using generalized cepstrum," *IEEE
           Transactions on Acoustics, Speech, and Signal Processing*, vol. 32, no. 5,
           pp. 1087-1089, 1984.

    """

    def __init__(self, cep_order: int, gamma: float = 0, c: int | None = None) -> None:
        super().__init__()

        self.in_dim = cep_order + 1
        self.values = self._precompute(**filter_values(locals()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform cepstrum gain normalization.

        Parameters
        ----------
        x : Tensor [shape=(..., M+1)]
            The generalized cepstrum.

        Returns
        -------
        out : Tensor [shape=(..., M+1)]
            The normalized generalized cepstrum.

        Examples
        --------
        >>> x = diffsptk.ramp(1, 4)
        >>> gnorm = diffsptk.GeneralizedCepstrumGainNormalization(3, c=2)
        >>> y = gnorm(x)
        >>> y
        tensor([2.2500, 1.3333, 2.0000, 2.6667])

        """
        check_size(x.size(-1), self.in_dim, "dimension of cepstrum")
        return self._forward(x, *self.values)

    @staticmethod
    def _func(x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        values = GeneralizedCepstrumGainNormalization._precompute(
            x.size(-1) - 1, *args, **kwargs
        )
        return GeneralizedCepstrumGainNormalization._forward(x, *values)

    @staticmethod
    def _takes_input_size() -> bool:
        return True

    @staticmethod
    def _check(cep_order: int, gamma: float, c: int | None) -> None:
        if cep_order < 0:
            raise ValueError("cep_order must be non-negative.")
        if 1 < abs(gamma):
            raise ValueError("gamma must be in [-1, 1].")
        if c is not None and c < 1:
            raise ValueError("c must be greater than or equal to 1.")

    @staticmethod
    def _precompute(cep_order: int, gamma: float, c: int | None = None) -> Precomputed:
        GeneralizedCepstrumGainNormalization._check(cep_order, gamma, c)
        return (get_gamma(gamma, c),)

    @staticmethod
    def _forward(x: torch.Tensor, gamma: float) -> torch.Tensor:
        x0, x1 = torch.split(x, [1, x.size(-1) - 1], dim=-1)
        if gamma == 0:
            K = torch.exp(x0)
            y = x1
        else:
            z = 1 + gamma * x0
            K = torch.pow(z, 1 / gamma)
            y = x1 / z
        y = torch.cat((K, y), dim=-1)
        return y
