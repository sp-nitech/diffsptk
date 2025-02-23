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

from .base import BaseFunctionalModule


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


class UniformQuantization(BaseFunctionalModule):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/quantize.html>`_
    for details. The gradient is copied from the subsequent module.

    Parameters
    ----------
    abs_max : float > 0
        The absolute maximum value of the input waveform.

    n_bit : int >= 1
        The number of quantization bits.

    quantizer : ['mid-rise', 'mid-tread']
        The quantizer type.

    """

    def __init__(self, abs_max=1, n_bit=8, quantizer="mid-rise"):
        super().__init__()

        self.values = self._precompute(abs_max, n_bit, quantizer)

    def forward(self, x):
        """Quantize the input waveform.

        Parameters
        ----------
        x : Tensor [shape=(...,)]
            The input waveform.

        Returns
        -------
        out : Tensor [shape=(...,)]
            The quantized waveform.

        Examples
        --------
        >>> x = diffsptk.ramp(-4, 4)
        >>> quantize = diffsptk.UniformQuantization(4, 2)
        >>> y = quantize(x).int()
        >>> y
        tensor([0, 0, 1, 1, 2, 2, 3, 3, 3], dtype=torch.int32)

        """
        return self._forward(x, *self.values)

    @staticmethod
    def _func(x, *args, **kwargs):
        values = UniformQuantization._precompute(*args, **kwargs)
        return UniformQuantization._forward(x, *values)

    @staticmethod
    def _check(abs_max, n_bit):
        if abs_max < 0:
            raise ValueError("abs_max must be non-negative.")
        if n_bit <= 0:
            raise ValueError("n_bit must be positive.")

    @staticmethod
    def _precompute(abs_max, n_bit, quantizer):
        UniformQuantization._check(abs_max, n_bit)
        if quantizer in (0, "mid-rise"):
            level = 1 << n_bit
            return (
                abs_max,
                level,
                lambda x: Floor.apply(x + level // 2),
            )
        elif quantizer in (1, "mid-tread"):
            level = (1 << n_bit) - 1
            return (
                abs_max,
                level,
                lambda x: Round.apply(x + (level - 1) // 2),
            )
        raise ValueError(f"quantizer {quantizer} is not supported.")

    @staticmethod
    def _forward(x, abs_max, level, func):
        y = func(x * (level / (2 * abs_max)))
        y = torch.clip(y, min=0, max=level - 1)
        return y
