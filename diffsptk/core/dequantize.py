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


class InverseUniformQuantization(nn.Module):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/dequantize.html>`_
    for details.

    Parameters
    ----------
    abs_max : float > 0 [scalar]
        Absolute maximum value of input.

    n_bit : int >= 1 [scalar]
        Number of quantization bits.

    quantizer : ['mid-rise', 'mid-tread']
        Quantizer.

    """

    def __init__(self, abs_max=1, n_bit=8, quantizer="mid-rise"):
        super(InverseUniformQuantization, self).__init__()

        self.abs_max = abs_max
        self.quantizer = quantizer

        assert 0 < self.abs_max
        assert 1 <= n_bit

        if quantizer == 0 or quantizer == "mid-rise":
            self.level = int(2**n_bit)
            self.quantizer = "mid-rise"
        elif quantizer == 1 or quantizer == "mid-tread":
            self.level = int(2**n_bit) - 1
            self.quantizer = "mid-tread"
        else:
            raise ValueError("quantizer {quantizer} is not supported")

    def forward(self, y):
        """Dequantize input.

        Parameters
        ----------
        y : Tensor [shape=(...,)]
            Quantized input.

        Returns
        -------
        x : Tensor [shape=(...,)]
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
        if self.quantizer == "mid-rise":
            y = y - (self.level // 2 - 0.5)
        elif self.quantizer == "mid-tread":
            y = y - (self.level - 1) // 2
        else:
            raise RuntimeError

        x = y * (2 * self.abs_max / self.level)
        x = torch.clip(x, min=-self.abs_max, max=self.abs_max)
        return x
