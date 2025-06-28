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

from ..typing import Callable, Precomputed
from ..utils.private import filter_values
from .base import BaseFunctionalModule
from .quantize import UniformQuantization


class InverseUniformQuantization(BaseFunctionalModule):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/dequantize.html>`_
    for details.

    Parameters
    ----------
    abs_max : float > 0
        The absolute maximum value of the original waveform.

    n_bit : int >= 1
        The number of quantization bits.

    quantizer : ['mid-rise', 'mid-tread']
        The quantizer type.

    """

    def __init__(
        self, abs_max: float = 1, n_bit: int = 8, quantizer: str | int = "mid-rise"
    ) -> None:
        super().__init__()

        self.values = self._precompute(**filter_values(locals()))

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """Dequantize the input waveform.

        Parameters
        ----------
        y : Tensor [shape=(...,)]
            The quantized waveform.

        Returns
        -------
        out : Tensor [shape=(...,)]
            The dequantized waveform.

        Examples
        --------
        >>> x = diffsptk.ramp(-4, 4)
        >>> x
        tensor([-4., -3., -2., -1.,  0.,  1.,  2.,  3.,  4.])
        >>> quantize = diffsptk.UniformQuantization(4, 2)
        >>> dequantize = diffsptk.InverseUniformQuantization(4, 2)
        >>> x2 = dequantize(quantize(x))
        >>> x2
        tensor([-3., -3., -1., -1.,  1.,  1.,  3.,  3.,  3.])

        """
        return self._forward(y, *self.values)

    @staticmethod
    def _func(y: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        values = InverseUniformQuantization._precompute(*args, **kwargs)
        return InverseUniformQuantization._forward(y, *values)

    @staticmethod
    def _takes_input_size() -> bool:
        return False

    @staticmethod
    def _check(*args, **kwargs) -> None:
        UniformQuantization._check(*args, **kwargs)

    @staticmethod
    def _precompute(abs_max: float, n_bit: int, quantizer: str | int) -> Precomputed:
        InverseUniformQuantization._check(abs_max, n_bit)
        if quantizer in (0, "mid-rise"):
            level = 1 << n_bit
            return (
                abs_max,
                level,
                lambda y: y - (level // 2 - 0.5),
            )
        elif quantizer in (1, "mid-tread"):
            level = (1 << n_bit) - 1
            return (
                abs_max,
                level,
                lambda y: y - (level // 2),
            )
        raise ValueError(f"quantizer {quantizer} is not supported.")

    @staticmethod
    def _forward(
        y: torch.Tensor, abs_max: float, level: int, func: Callable
    ) -> torch.Tensor:
        x = func(y) * (2 * abs_max / level)
        x = torch.clip(x, min=-abs_max, max=abs_max)
        return x
