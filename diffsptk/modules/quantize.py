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

from ..misc.utils import check

_quantization_level = {
    "mid-rise": lambda n_bit: 1 << n_bit,
    "mid-tread": lambda n_bit: (1 << n_bit) - 1,
}


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
        self.n_bit = n_bit
        self.quantizer = check(quantizer, _quantization_level)

        assert 0 < self.abs_max
        assert 1 <= self.n_bit

    def forward(self, x):
        """Quantize input.

        Parameters
        ----------
        x : Tensor [shape=(...,)]
            Input.

        Returns
        -------
        Tensor [shape=(...,)]
            Quantized input.

        Examples
        --------
        >>> x = diffsptk.ramp(-4, 4)
        >>> quantize = diffsptk.UniformQuantization(4, 2)
        >>> y = quantize(x).int()
        >>> y
        tensor([0, 0, 1, 1, 2, 2, 3, 3, 3], dtype=torch.int32)

        """
        return self._forward(x, self.abs_max, self.n_bit, self.quantizer)

    @staticmethod
    def _forward(x, abs_max, n_bit, quantizer):
        try:
            level = _quantization_level[quantizer](n_bit)
        except KeyError:
            raise ValueError(f"quantizer {quantizer} is not supported")

        x = x * (level / (2 * abs_max))
        if quantizer == "mid-rise":
            x += level // 2
            y = Floor.apply(x)
        elif quantizer == "mid-tread":
            x += (level - 1) // 2
            y = Round.apply(x)
        else:
            raise ValueError(f"quantizer {quantizer} is not supported")
        y = torch.clip(y, min=0, max=level - 1)
        return y
