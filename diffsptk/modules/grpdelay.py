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
import torch.nn.functional as F

from ..typing import Precomputed
from ..utils.private import filter_values, remove_gain
from .base import BaseFunctionalModule


class GroupDelay(BaseFunctionalModule):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/grpdelay.html>`_
    for details.

    Parameters
    ----------
    fft_length : int >= 2
        The number of FFT bins, :math:`L`.

    alpha : float > 0
        The tuning parameter, :math:`\\alpha`.

    gamma : float > 0
        The tuning parameter, :math:`\\gamma`.

    device : torch.device or None
        The device of this module.

    dtype : torch.dtype or None
        The data type of this module.

    """

    def __init__(
        self,
        fft_length: int,
        alpha: float = 1,
        gamma: float = 1,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        self.values, _, tensors = self._precompute(**filter_values(locals()))
        self.register_buffer("ramp", tensors[0])

    def forward(
        self, b: torch.Tensor | None = None, a: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Compute group delay.

        Parameters
        ----------
        b : Tensor [shape=(..., M+1)] or None
            The numerator coefficients.

        a : Tensor [shape=(..., N+1)] or None
            The denominator coefficients.

        Returns
        -------
        out : Tensor [shape=(..., L/2+1)]
            The group delay or modified group delay function.

        Examples
        --------
        >>> x = diffsptk.ramp(3)
        >>> grpdelay = diffsptk.GroupDelay(8)
        >>> g = grpdelay(x)
        >>> g
        tensor([2.3333, 2.4278, 3.0000, 3.9252, 3.0000])

        """
        return self._forward(b, a, *self.values, **self._buffers)

    @staticmethod
    def _func(
        b: torch.Tensor | None, a: torch.Tensor | None, *args, **kwargs
    ) -> torch.Tensor:
        x = a if b is None else b
        values, _, tensors = GroupDelay._precompute(
            *args, **kwargs, device=x.device, dtype=x.dtype
        )
        return GroupDelay._forward(b, a, *values, *tensors)

    @staticmethod
    def _takes_input_size() -> bool:
        return False

    @staticmethod
    def _check(fft_length: int, alpha: float, gamma: float) -> None:
        if fft_length <= 1:
            raise ValueError("fft_length must be greater than 1.")
        if alpha <= 0:
            raise ValueError("alpha must be positive.")
        if gamma <= 0:
            raise ValueError("gamma must be positive.")

    @staticmethod
    def _precompute(
        fft_length: int,
        alpha: float,
        gamma: float,
        device: torch.device | None,
        dtype: torch.dtype | None,
    ) -> Precomputed:
        GroupDelay._check(fft_length, alpha, gamma)
        ramp = torch.arange(fft_length, device=device, dtype=dtype)
        return (fft_length, alpha, gamma), None, (ramp,)

    @staticmethod
    def _forward(
        b: torch.Tensor | None,
        a: torch.Tensor | None,
        fft_length: int,
        alpha: float,
        gamma: float,
        ramp: torch.Tensor,
    ) -> torch.Tensor:
        if b is None and a is None:
            raise ValueError("Either b or a must be specified.")

        if a is None:
            order = 0
        else:
            a = remove_gain(a)
            order = a.size(-1) - 1

        if b is None:
            c = a.flip(-1)
        elif a is None:
            c = b
        else:
            # Perform full convolution.
            b1 = F.pad(b, (order, order))
            b2 = b1.unfold(-1, b.size(-1) + order, 1)
            c = (b2 * a.unsqueeze(-1)).sum(-2)

        data_length = c.size(-1)
        if fft_length < data_length:
            raise RuntimeError("Please increase FFT length.")

        d = c * ramp[:data_length]
        C = torch.fft.rfft(c, n=fft_length)
        D = torch.fft.rfft(d, n=fft_length)

        numer = C.real * D.real + C.imag * D.imag
        denom = C.real * C.real + C.imag * C.imag
        if gamma != 1:
            denom = torch.pow(denom, gamma)

        g = numer / denom - order
        if alpha != 1:
            g = torch.sign(g) * torch.pow(torch.abs(g), alpha)
        return g
