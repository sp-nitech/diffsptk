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
import torch.nn as nn


class MuLawExpansion(nn.Module):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/iulaw.html>`_
    for details.

    Parameters
    ----------
    abs_max : float > 0 [scalar]
        Absolute maximum value of input.

    mu : int >= 1 [scalar]
        Compression factor, :math:`\\mu`.

    """

    def __init__(self, abs_max=1, mu=255):
        super(MuLawExpansion, self).__init__()

        self.abs_max = abs_max
        self.mu = mu

        assert 0 < self.abs_max
        assert 1 <= self.mu

        self.const = self.abs_max / self.mu

    def forward(self, y):
        """Expand waveform by :math:`\\mu`-law algorithm.

        Parameters
        ----------
        y : Tensor [shape=(...,)]
            Compressed waveform.

        Returns
        -------
        x : Tensor [shape=(...,)]
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
        y_abs = y.abs() / self.abs_max
        x = self.const * torch.sign(y) * (torch.pow(1 + self.mu, y_abs) - 1)
        return x
