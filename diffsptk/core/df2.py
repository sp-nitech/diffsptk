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

import torch.nn as nn

from .dfs import InfiniteImpulseResponseDigitalFilter


class SecondOrderDigitalFilter(nn.Module):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/df2.html>`_
    for details.

    Parameters
    ----------
    sample_rate : int >= 1 [scalar]
        Sample rate in Hz.

    pole_frequency : float > 0 [scalar]
        Pole frequency in Hz.

    pole_bandwidth : float > 0 [scalar]
        Pole bandwidth in Hz.

    zero_frequency : float > 0 [scalar]
        Zero frequency in Hz.

    zero_bandwidth : float > 0 [scalar]
        Zero bandwidth in Hz.

    ir_length : int >= 1 [scalar]
        Length of impulse response.

    """

    def __init__(
        self,
        sample_rate,
        pole_frequency=None,
        pole_bandwidth=None,
        zero_frequency=None,
        zero_bandwidth=None,
        ir_length=None,
    ):
        super(SecondOrderDigitalFilter, self).__init__()

        def get_filter_coefficients(frequency, bandwidth, sample_rate):
            assert 0 < frequency
            assert 0 < bandwidth
            assert 0 < sample_rate
            assert frequency <= sample_rate / 2
            r = math.exp(-math.pi * bandwidth / sample_rate)
            theta = 2 * math.pi * frequency / sample_rate
            return [1, -2 * r * math.cos(theta), r * r]

        param = {}
        if pole_frequency is not None:
            param["a"] = get_filter_coefficients(
                pole_frequency, pole_bandwidth, sample_rate
            )
        if zero_frequency is not None:
            param["b"] = get_filter_coefficients(
                zero_frequency, zero_bandwidth, sample_rate
            )
        if ir_length is not None:
            param["ir_length"] = ir_length

        self.dfs = InfiniteImpulseResponseDigitalFilter(**param)

    def forward(self, x):
        """Apply a second order digital filter.

        Parameters
        ----------
        x : Tensor [shape=(B, 1, T) or (B, T) or (T,)]
            Input waveform.

        Returns
        -------
        y : Tensor [shape=(B, 1, T) or (B, T) or (T,)]
            Filterd waveform.

        Examples
        --------
        >>> x = diffsptk.ramp(4)
        >>> df2 = diffsptk.SecondOrderDigitalFilter(16000, 100, 200, 1000, 50, 5)
        >>> y = df2(x)
        >>> y
        tensor([0.0000, 1.0000, 2.0918, 3.4161, 5.1021])

        """
        y = self.dfs(x)
        return y
