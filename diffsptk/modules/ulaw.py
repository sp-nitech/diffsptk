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


class MuLawCompression(nn.Module):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/ulaw.html>`_
    for details.

    Parameters
    ----------
    abs_max : float > 0
        Absolute maximum value of input.

    mu : int >= 1
        Compression factor, :math:`\\mu`.

    """

    def __init__(self, abs_max=1, mu=255):
        super().__init__()

        assert 0 < abs_max
        assert 1 <= mu

        self.abs_max = abs_max
        self.mu = mu
        self.const = self._precompute(self.abs_max, self.mu)

    def forward(self, x):
        """Compress waveform by :math:`\\mu`-law algorithm.

        Parameters
        ----------
        x : Tensor [shape=(...,)]
            Waveform.

        Returns
        -------
        out : Tensor [shape=(...,)]
            Compressed waveform.

        Examples
        --------
        >>> x = diffsptk.ramp(4)
        >>> ulaw = diffsptk.MuLawCompression(4)
        >>> y = ulaw(x)
        >>> y
        tensor([0.0000, 3.0084, 3.5028, 3.7934, 4.0000])

        """
        return self._forward(x, self.abs_max, self.mu, self.const)

    @staticmethod
    def _forward(x, abs_max, mu, const):
        x_abs = x.abs() / abs_max
        y = const * torch.sign(x) * torch.log1p(mu * x_abs)
        return y

    @staticmethod
    def _func(x, abs_max, mu):
        const = MuLawCompression._precompute(abs_max, mu)
        return MuLawCompression._forward(x, abs_max, mu, const)

    @staticmethod
    def _precompute(abs_max, mu):
        return abs_max / math.log1p(mu)
