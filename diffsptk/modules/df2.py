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

import math

import torch
from torch import nn

from ..typing import Callable, Precomputed
from ..utils.private import filter_values
from .base import BaseNonFunctionalModule
from .dfs import InfiniteImpulseResponseDigitalFilter


class SecondOrderDigitalFilter(BaseNonFunctionalModule):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/df2.html>`_
    for details.

    Parameters
    ----------
    sample_rate : int >= 1
        The sample rate in Hz.

    pole_frequency : float > 0 or None
        The pole frequency in Hz.

    pole_bandwidth : float > 0 or None
        The pole bandwidth in Hz.

    zero_frequency : float > 0 or None
        The zero frequency in Hz.

    zero_bandwidth : float > 0 or None
        The zero bandwidth in Hz.

    ir_length : int >= 1 or None
        The length of the truncated impulse response. If given, the filter is
        approximated by an FIR filter.

    device : torch.device or None
        The device of this module.

    dtype : torch.dtype or None
        The data type of this module.

    """

    def __init__(
        self,
        sample_rate: int,
        pole_frequency: float | None = None,
        pole_bandwidth: float | None = None,
        zero_frequency: float | None = None,
        zero_bandwidth: float | None = None,
        ir_length: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        _, layers, _ = self._precompute(**filter_values(locals()))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply a second order digital filter to the input waveform.

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
        >>> df2 = diffsptk.SecondOrderDigitalFilter(16000, 100, 200, 1000, 50, 5)
        >>> y = df2(x)
        >>> y
        tensor([0.0000, 1.0000, 2.0918, 3.4161, 5.1021])

        """
        return self._forward(x, *self.layers)

    @staticmethod
    def _check(
        sample_rate: int,
        pole_frequency: float | None,
        pole_bandwidth: float | None,
        zero_frequency: float | None,
        zero_bandwidth: float | None,
    ) -> None:
        if pole_frequency is not None and pole_frequency <= 0:
            raise ValueError("pole_frequency must be positive.")
        if pole_bandwidth is not None and pole_bandwidth <= 0:
            raise ValueError("pole_bandwidth must be positive.")
        if zero_frequency is not None and zero_frequency <= 0:
            raise ValueError("zero_frequency must be positive.")
        if zero_bandwidth is not None and zero_bandwidth <= 0:
            raise ValueError("zero_bandwidth must be positive.")
        nyquist = sample_rate / 2
        if pole_frequency is not None and nyquist < pole_frequency:
            raise ValueError("pole_frequency must be less than Nyquist frequency.")
        if zero_frequency is not None and nyquist < zero_frequency:
            raise ValueError("zero_frequency must be less than Nyquist frequency.")

    @staticmethod
    def _precompute(
        sample_rate: int,
        pole_frequency: float | None,
        pole_bandwidth: float | None,
        zero_frequency: float | None,
        zero_bandwidth: float | None,
        ir_length: int | None,
        device: torch.device | None,
        dtype: torch.dtype | None,
    ) -> Precomputed:
        SecondOrderDigitalFilter._check(
            sample_rate, pole_frequency, pole_bandwidth, zero_frequency, zero_bandwidth
        )

        def get_filter_coefficients(sample_rate, frequency, bandwidth):
            r = math.exp(-math.pi * bandwidth / sample_rate)
            theta = math.tau * frequency / sample_rate
            return [1, -2 * r * math.cos(theta), r * r]

        a = b = None
        if pole_frequency is not None:
            a = get_filter_coefficients(sample_rate, pole_frequency, pole_bandwidth)
        if zero_frequency is not None:
            b = get_filter_coefficients(sample_rate, zero_frequency, zero_bandwidth)
        dfs = InfiniteImpulseResponseDigitalFilter(
            a=a, b=b, ir_length=ir_length, device=device, dtype=dtype
        )
        return None, (dfs,), None

    @staticmethod
    def _forward(x: torch.Tensor, dfs: Callable) -> torch.Tensor:
        return dfs(x)
