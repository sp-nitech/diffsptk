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

from ..misc.utils import check_mode
from .quantize import _quantization_level


class InverseUniformQuantization(nn.Module):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/dequantize.html>`_
    for details.

    Parameters
    ----------
    abs_max : float > 0
        Absolute maximum value of input.

    n_bit : int >= 1
        Number of quantization bits.

    quantizer : ['mid-rise', 'mid-tread']
        Quantizer.

    """

    def __init__(self, abs_max=1, n_bit=8, quantizer="mid-rise"):
        super(InverseUniformQuantization, self).__init__()

        self.abs_max = abs_max
        self.n_bit = n_bit
        self.quantizer = check_mode(quantizer, _quantization_level)

        assert 0 < self.abs_max
        assert 1 <= self.n_bit

    def forward(self, y):
        """Dequantize input.

        Parameters
        ----------
        y : Tensor [shape=(...,)]
            Quantized input.

        Returns
        -------
        Tensor [shape=(...,)]
            Dequantized input.

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
        return self._forward(y, self.abs_max, self.n_bit, self.quantizer)

    @staticmethod
    def _forward(y, abs_max, n_bit, quantizer):
        try:
            level = _quantization_level[quantizer](n_bit)
        except KeyError:
            raise ValueError(f"quantizer {quantizer} is not supported")

        if quantizer == "mid-rise":
            y = y - (level // 2 - 0.5)
        elif quantizer == "mid-tread":
            y = y - (level - 1) // 2
        else:
            raise ValueError(f"quantizer {quantizer} is not supported")
        x = y * (2 * abs_max / level)
        x = torch.clip(x, min=-abs_max, max=abs_max)
        return x
