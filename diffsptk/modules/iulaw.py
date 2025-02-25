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

from ..misc.utils import get_values
from .base import BaseFunctionalModule
from .ulaw import MuLawCompression


class MuLawExpansion(BaseFunctionalModule):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/iulaw.html>`_
    for details.

    Parameters
    ----------
    abs_max : float > 0
        The absolute maximum value of the original input waveform.

    mu : int >= 1
        The compression factor, :math:`\\mu`.

    """

    def __init__(self, abs_max=1, mu=255):
        super().__init__()

        self.values = self._precompute(*get_values(locals()))

    def forward(self, y):
        """Expand the waveform using the :math:`\\mu`-law algorithm.

        Parameters
        ----------
        y : Tensor [shape=(...,)]
            The input compressed waveform.

        Returns
        -------
        out : Tensor [shape=(...,)]
            The expanded waveform.

        Examples
        --------
        >>> x = diffsptk.ramp(4)
        >>> ulaw = diffsptk.MuLawCompression(4)
        >>> iulaw = diffsptk.MuLawExpansion(4)
        >>> x2 = iulaw(ulaw(x))
        >>> x2
        tensor([0.0000, 1.0000, 2.0000, 3.0000, 4.0000])

        """
        return self._forward(y, *self.values)

    @staticmethod
    def _func(y, *args, **kwargs):
        values = MuLawExpansion._precompute(*args, **kwargs)
        return MuLawExpansion._forward(y, *values)

    @staticmethod
    def _check(*args, **kwargs):
        MuLawCompression._check(*args, **kwargs)

    @staticmethod
    def _precompute(abs_max, mu):
        MuLawExpansion._check(abs_max, mu)
        return (
            abs_max,
            mu,
            abs_max / mu,
        )

    @staticmethod
    def _forward(y, abs_max, mu, c):
        y_abs = y.abs() / abs_max
        x = c * torch.sign(y) * (torch.pow(1 + mu, y_abs) - 1)
        return x
