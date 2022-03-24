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
import torch.nn as nn


class MuLawCompression(nn.Module):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/ulaw.html>`_
    for details.

    Parameters
    ----------
    abs_max : float > 0 [scalar]
        Absolute maximum value of input.

    mu : int >= 1 [scalar]
        Compression factor, :math:`\\mu`.

    """

    def __init__(self, abs_max=1, mu=255):
        super(MuLawCompression, self).__init__()

        self.abs_max = abs_max
        self.mu = mu

        assert 0 < self.abs_max
        assert 0 < self.mu

        self.const = self.abs_max / math.log1p(self.mu)

    def forward(self, x):
        """Compress waveform by :math:`\\mu`-law algorithm.

        Parameters
        ----------
        x : Tensor [shape=(...,)]
            Waveform.

        Returns
        -------
        y : Tensor [shape=(...,)]
            Compressed waveform.

        Examples
        --------
        >>> x = diffsptk.ramp(4)
        >>> ulaw = diffsptk.MuLawCompression(4)
        >>> y = ulaw(x)
        >>> y
        tensor([0.0000, 3.0084, 3.5028, 3.7934, 4.0000])

        """
        x2 = torch.abs(x) / self.abs_max
        y = self.const * torch.sign(x) * torch.log1p(self.mu * x2)
        return y
