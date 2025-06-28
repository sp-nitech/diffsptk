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

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ..typing import ArrayLike, Precomputed
from ..utils.private import filter_values, iir, to, to_3d
from .base import BaseFunctionalModule


class InfiniteImpulseResponseDigitalFilter(BaseFunctionalModule):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/dfs.html>`_
    for details.

    Parameters
    ----------
    b : List [shape=(M+1,)] or None
        The numerator coefficients.

    a : List [shape=(N+1,)] or None
        The denominator coefficients.

    ir_length : int >= 1 or None
        The length of the truncated impulse response. If given, the filter is
        approximated by an FIR filter.

    learnable : bool
        If True, the filter coefficients are learnable.

    device : torch.device or None
        The device of this module.

    dtype : torch.dtype or None
        The data type of this module.

    """

    def __init__(
        self,
        b: ArrayLike | None = None,
        a: ArrayLike | None = None,
        ir_length: int | None = None,
        learnable: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        _, _, tensors = self._precompute(
            **filter_values(locals(), drop_keys=["learnable"])
        )
        if learnable and b is not None:
            self.b = nn.Parameter(tensors[0])
        else:
            self.register_buffer("b", tensors[0])
        if learnable and a is not None:
            self.a = nn.Parameter(tensors[1])
        else:
            self.register_buffer("a", tensors[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply an IIR digital filter to the input waveform.

        Parameters
        ----------
        x : Tensor [shape=(..., T)]
            The input waveform.

        Returns
        -------
        out : Tensor [shape=(..., T)]
            The filtered waveform.

        Examples
        --------
        >>> x = diffsptk.ramp(4)
        >>> dfs = diffsptk.IIR(b=[1, -0.97])
        >>> y = dfs(x)
        >>> y
        tensor([0.0000, 1.0000, 1.0300, 1.0600, 1.0900])

        """
        return self._forward(x, **self._buffers, **self._parameters)

    @staticmethod
    def _func(x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        _, _, tensors = InfiniteImpulseResponseDigitalFilter._precompute(
            *args, **kwargs, device=x.device, dtype=x.dtype
        )
        return InfiniteImpulseResponseDigitalFilter._forward(x, *tensors)

    @staticmethod
    def _takes_input_size() -> bool:
        return False

    @staticmethod
    def _check(ir_length: int | None) -> None:
        if ir_length is not None and ir_length <= 0:
            raise ValueError("ir_length must be positive.")

    @staticmethod
    def _precompute(
        b: ArrayLike | None,
        a: ArrayLike | None,
        ir_length: int | None,
        device: torch.device | None,
        dtype: torch.dtype | None,
    ) -> Precomputed:
        InfiniteImpulseResponseDigitalFilter._check(ir_length)

        fir = a is None

        if b is None:
            b = [1]
        if a is None:
            a = [1]
        if not torch.is_tensor(b):
            b = to(np.asarray(b), device=device, dtype=dtype)
        if not torch.is_tensor(a):
            a = to(np.asarray(a), device=device, dtype=dtype)

        if fir:
            b = b.view(1, 1, -1).flip(-1)
            a = torch.empty(0)
        elif ir_length is not None:
            # Pre-compute the truncated impulse response.
            d = torch.zeros(max(len(b), len(a)), device=device, dtype=torch.double)
            h = torch.empty(ir_length, device=device, dtype=torch.double)
            a0 = a[0]
            a1 = a[1:]
            for t in range(ir_length):
                x = a0 if t == 0 else 0
                y = x - torch.sum(d[: len(a1)] * a1)
                d = torch.roll(d, 1)
                d[0] = y
                y = torch.sum(d[: len(b)] * b)
                h[t] = y
            h = h.view(1, 1, -1).flip(-1)
            b = to(h, dtype=dtype)
            a = torch.empty(0)
        return None, None, (b, a)

    @staticmethod
    def _forward(x: torch.Tensor, b: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        if len(a) == 0:
            y = to_3d(x)
            y = F.pad(y, (b.size(-1) - 1, 0))
            y = F.conv1d(y, b)
            y = y.view_as(x)
        else:
            y = iir(x, b, a)
        return y
