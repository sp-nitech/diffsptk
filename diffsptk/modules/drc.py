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
import torchcomp
from torch import nn

from ..typing import Precomputed
from ..utils.private import get_values, to, to_2d
from .base import BaseFunctionalModule


class DynamicRangeCompression(BaseFunctionalModule):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/drc.html>`_
    for details.

    Parameters
    ----------
    threshold : float <= 0
        The threshold in dB.

    ratio : float > 1
        The input/output ratio.

    attack_time : float > 0
        The attack time in msec.

    release_time : float > 0
        The release time in msec.

    sample_rate : int >= 1
        The sample rate in Hz.

    makeup_gain : float >= 0
        The make-up gain in dB.

    abs_max : float > 0
        The absolute maximum value of input.

    learnable : bool
        Whether to make the DRC parameters learnable.

    device : torch.device or None
        The device of this module.

    dtype : torch.dtype or None
        The data type of this module.

    References
    ----------
    .. [1] C.-Y. Yu et al., "Differentiable all-pole filters for time-varying audio
           systems," *Proceedings of DAFx*, pp. 345-352, 2024.

    """

    def __init__(
        self,
        threshold: float,
        ratio: float,
        attack_time: float,
        release_time: float,
        sample_rate: int,
        makeup_gain: float = 0,
        abs_max: float = 1,
        learnable: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        self.values, _, tensors = self._precompute(
            *get_values(locals(), drop_keys=["learnable"])
        )
        if learnable:
            self.params = nn.Parameter(tensors[0])
        else:
            self.register_buffer("params", tensors[0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform dynamic range compression.

        Parameters
        ----------
        x : Tensor [shape=(..., T)]
            The input waveform.

        Returns
        -------
        out : Tensor [shape=(..., T)]
            The compressed waveform.

        Examples
        --------
        >>> x = torch.randn(16000)
        >>> x.abs().max()
        tensor(4.2224)
        >>> drc = diffsptk.DynamicRangeCompression(-20, 4, 10, 100, 16000)
        >>> y = drc(x)
        >>> y.abs().max()
        tensor(2.5779)

        """
        return self._forward(x, *self.values, **self._buffers, **self._parameters)

    @staticmethod
    def _func(x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        values, _, tensors = DynamicRangeCompression._precompute(
            *args, **kwargs, device=x.device, dtype=x.dtype
        )
        return DynamicRangeCompression._forward(x, *values, *tensors)

    @staticmethod
    def _takes_input_size() -> bool:
        return False

    @staticmethod
    def _check(
        ratio: float,
        attack_time: float,
        release_time: float,
        sample_rate: int,
        makeup_gain: float,
        abs_max: float,
    ) -> None:
        if ratio <= 1:
            raise ValueError("ratio must be greater than 1.")
        if attack_time <= 0:
            raise ValueError("attack_time must be positive.")
        if release_time <= 0:
            raise ValueError("release_time must be positive.")
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive.")
        if makeup_gain < 0:
            raise ValueError("makeup_gain must be non-negative.")
        if abs_max <= 0:
            raise ValueError("abs_max must be positive.")

    @staticmethod
    def _precompute(
        threshold: float,
        ratio: float,
        attack_time: float,
        release_time: float,
        sample_rate: int,
        makeup_gain: float,
        abs_max: float,
        device: torch.device | None,
        dtype: torch.dtype | None,
    ) -> Precomputed:
        DynamicRangeCompression._check(
            ratio, attack_time, release_time, sample_rate, makeup_gain, abs_max
        )
        c = round(np.log(9), 1)
        if not torch.is_tensor(threshold):
            threshold = to(torch.tensor(threshold, device=device), dtype=dtype)
        if not torch.is_tensor(ratio):
            ratio = to(torch.tensor(ratio, device=device), dtype=dtype)
        if not torch.is_tensor(attack_time):
            attack_time = to(torch.tensor(attack_time, device=device), dtype=dtype)
        attack_time = torchcomp.ms2coef(attack_time * c, sample_rate)
        if not torch.is_tensor(release_time):
            release_time = to(torch.tensor(release_time, device=device), dtype=dtype)
        release_time = torchcomp.ms2coef(release_time * c, sample_rate)
        if not torch.is_tensor(makeup_gain):
            makeup_gain = to(torch.tensor(makeup_gain, device=device), dtype=dtype)
        makeup_gain = 10 ** (makeup_gain / 20)
        params = torch.stack([threshold, ratio, attack_time, release_time, makeup_gain])
        return (abs_max,), None, (params,)

    @staticmethod
    def _forward(
        x: torch.Tensor,
        abs_max: float,
        params: torch.Tensor,
    ) -> torch.Tensor:
        eps = 1e-10

        y = to_2d(x)
        y_abs = y.abs() / abs_max + eps

        g = torchcomp.compexp_gain(
            y_abs,
            params[0],
            params[1],
            -1000,  # Expander threshold
            eps,  # Expander ratio
            params[2],
            params[3],
        )

        makeup_gain = params[4]
        y = y * g * makeup_gain
        y = y.view_as(x)
        return y
