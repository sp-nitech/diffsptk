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
from torch import nn


class MuLawExpansion(nn.Module):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/iulaw.html>`_
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

    def forward(self, y):
        """Expand waveform by :math:`\\mu`-law algorithm.

        Parameters
        ----------
        y : Tensor [shape=(...,)]
            Compressed waveform.

        Returns
        -------
        out : Tensor [shape=(...,)]
            Waveform.

        Examples
        --------
        >>> x = diffsptk.ramp(4)
        >>> ulaw = diffsptk.MuLawCompression(4)
        >>> iulaw = diffsptk.MuLawExpansion(4)
        >>> x2 = iulaw(ulaw(x))
        >>> x2
        tensor([0.0000, 1.0000, 2.0000, 3.0000, 4.0000])

        """
        return self._forward(y, self.abs_max, self.mu, self.const)

    @staticmethod
    def _forward(y, abs_max, mu, const):
        y_abs = y.abs() / abs_max
        x = const * torch.sign(y) * (torch.pow(1 + mu, y_abs) - 1)
        return x

    @staticmethod
    def _func(y, abs_max, mu):
        const = MuLawExpansion._precompute(abs_max, mu)
        return MuLawExpansion._forward(y, abs_max, mu, const)

    @staticmethod  # noqa: FURB118
    def _precompute(abs_max, mu):
        return abs_max / mu
