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
from torch import nn
import torchcomp

from ..misc.utils import to_2d


class DynamicRangeCompression(nn.Module):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/drc.html>`_
    for details.

    Parameters
    ----------
    threshold : float <= 0
        Threshold in dB.

    ratio : float > 1
        Input/output ratio.

    attack_time : float > 0
        Attack time in msec.

    release_time : float > 0
        Release time in msec.

    sample_rate : int >= 1
        Sample rate in Hz.

    makeup_gain : float >= 0
        Make-up gain in dB.

    abs_max : float > 0
        Absolute maximum value of input.

    learnable : bool
        Whether to make the DRC parameters learnable.

    References
    ----------
    .. [1] C.-Y. Yu et al., "Differentiable all-pole filters for time-varying audio
           systems," *Proceedings of DAFx*, 2024.

    """

    def __init__(
        self,
        threshold,
        ratio,
        attack_time,
        release_time,
        sample_rate,
        makeup_gain=0,
        abs_max=1,
        learnable=False,
    ):
        super().__init__()

        assert threshold <= 0
        assert 1 < ratio
        assert 0 < attack_time
        assert 0 < release_time
        assert 1 <= sample_rate
        assert 0 <= makeup_gain
        assert 0 < abs_max

        self.abs_max = abs_max
        params = self._precompute(
            threshold, ratio, attack_time, release_time, sample_rate, makeup_gain
        )
        if learnable:
            self.params = nn.Parameter(params)
        else:
            self.register_buffer("params", params)

    def forward(self, x):
        """Perform dynamic range compression.

        Parameters
        ----------
        x : Tensor [shape=(..., T)]
            Input signal.

        Returns
        -------
        out : Tensor [shape=(..., T)]
            Compressed signal.

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
        return self._forward(x, self.abs_max, self.params)

    @staticmethod
    def _forward(x, abs_max, params):
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

        makeup_gain = params[-1]
        y = y * g * makeup_gain
        y = y.view_as(x)
        return y

    @staticmethod
    def _func(
        x,
        threshold,
        ratio,
        attack_time,
        release_time,
        sample_rate,
        makeup_gain,
        abs_max,
    ):
        params = DynamicRangeCompression._precompute(
            threshold,
            ratio,
            attack_time,
            release_time,
            sample_rate,
            makeup_gain,
            dtype=x.dtype,
            device=x.device,
        )
        return DynamicRangeCompression._forward(x, abs_max, params)

    @staticmethod
    def _precompute(
        threshold,
        ratio,
        attack_time,
        release_time,
        sample_rate,
        makeup_gain,
        dtype=None,
        device=None,
    ):
        c = round(np.log(9), 1)
        attack_time = (
            torchcomp.ms2coef(torch.tensor(attack_time * c), sample_rate).cpu().numpy()
        )
        release_time = (
            torchcomp.ms2coef(torch.tensor(release_time * c), sample_rate).cpu().numpy()
        )
        makeup_gain = 10 ** (makeup_gain / 20)
        params = np.array([threshold, ratio, attack_time, release_time, makeup_gain])
        return torch.tensor(params, dtype=dtype, device=device)
