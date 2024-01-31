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


class Floor(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.floor()

    @staticmethod
    def backward(ctx, grad):
        return grad


class Round(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return (x + 0.5 * torch.sign(x)).trunc()

    @staticmethod
    def backward(ctx, grad):
        return grad


class UniformQuantization(nn.Module):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/quantize.html>`_
    for details. The gradient is copied from the next module.

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
        super(UniformQuantization, self).__init__()

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

    def forward(self, x):
        """Quantize input.

        Parameters
        ----------
        x : Tensor [shape=(...,)]
            Input.

        Returns
        -------
        y : Tensor [shape=(...,)]
            Quantized input.

        Examples
        --------
        >>> x = diffsptk.ramp(-4, 4)
        >>> quantize = diffsptk.UniformQuantization(4, 2)
        >>> y = quantize(x).int()
        >>> y
        tensor([0, 0, 1, 1, 2, 2, 3, 3, 3], dtype=torch.int32)

        """
        x = x * self.level / (2 * self.abs_max)
        if self.quantizer == "mid-rise":
            x = x + self.level // 2
            y = Floor.apply(x)
        elif self.quantizer == "mid-tread":
            x = x + (self.level - 1) // 2
            y = Round.apply(x)
        else:
            raise RuntimeError

        y = torch.clip(y, min=0, max=self.level - 1)
        return y
