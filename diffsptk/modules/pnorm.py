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

import inspect

import torch
from torch import nn

from ..typing import Callable, Precomputed
from ..utils.private import get_layer, get_values
from .base import BaseFunctionalModule
from .c2acr import CepstrumToAutocorrelation
from .freqt import FrequencyTransform


class MelCepstrumPowerNormalization(BaseFunctionalModule):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/pnorm.html>`_
    for details.

    Parameters
    ----------
    cep_order : int >= 0
        The order of the cepstrum, :math:`M`.

    alpha : float in (-1, 1)
        The frequency warping factor, :math:`\\alpha`.

    ir_length : int >= 1
        The length of the impulse response.

    """

    def __init__(self, cep_order: int, alpha: float = 0, ir_length: int = 128) -> None:
        super().__init__()

        _, layers, _ = self._precompute(*get_values(locals()))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform mel-cepstrum power normalization.

        Parameters
        ----------
        x : Tensor [shape=(..., M+1)]
            The input mel-cepstrum.

        Returns
        -------
        out : Tensor [shape=(..., M+2)]
            The log power and power-normalized mel-cepstrum.

        Examples
        --------
        >>> x = diffsptk.ramp(1, 4)
        >>> pnorm = diffsptk.MelCepstrumPowerNormalization(3, alpha=0.1)
        >>> y = pnorm(x)
        >>> y
        tensor([ 8.2942, -7.2942,  2.0000,  3.0000,  4.0000])

        """
        return self._forward(x, *self.layers)

    @staticmethod
    def _func(x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        _, layers, _ = MelCepstrumPowerNormalization._precompute(
            x.size(-1) - 1, *args, **kwargs, device=x.device, dtype=x.dtype
        )
        return MelCepstrumPowerNormalization._forward(x, *layers)

    @staticmethod
    def _takes_input_size() -> bool:
        return True

    @staticmethod
    def _check() -> None:
        pass

    @staticmethod
    def _precompute(
        cep_order: int,
        alpha: float,
        ir_length: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Precomputed:
        MelCepstrumPowerNormalization._check()
        module = inspect.stack()[1].function == "__init__"

        freqt = get_layer(
            module,
            FrequencyTransform,
            dict(
                in_order=cep_order,
                out_order=ir_length - 1,
                alpha=-alpha,
            ),
        )
        c2acr = get_layer(
            module,
            CepstrumToAutocorrelation,
            dict(
                cep_order=ir_length - 1,
                acr_order=0,
                n_fft=ir_length,
            ),
        )
        return None, (freqt, c2acr), None

    @staticmethod
    def _forward(x: torch.Tensor, freqt: Callable, c2acr: Callable) -> torch.Tensor:
        x0, x1 = torch.split(x, [1, x.size(-1) - 1], dim=-1)
        P = torch.log(c2acr(freqt(x)))
        y = torch.cat((P, x0 - 0.5 * P, x1), dim=-1)
        return y
