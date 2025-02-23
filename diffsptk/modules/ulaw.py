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

from .base import BaseFunctionalModule


class MuLawCompression(BaseFunctionalModule):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/ulaw.html>`_
    for details.

    Parameters
    ----------
    abs_max : float > 0
        The absolute maximum value of the input waveform.

    mu : int >= 1
        The compression factor, :math:`\\mu`.

    """

    def __init__(self, abs_max=1, mu=255):
        super().__init__()

        self.values = self._precompute(abs_max, mu)

    def forward(self, x):
        """Compress the input waveform using the :math:`\\mu`-law algorithm.

        Parameters
        ----------
        x : Tensor [shape=(...,)]
            The input waveform.

        Returns
        -------
        out : Tensor [shape=(...,)]
            The compressed waveform.

        Examples
        --------
        >>> x = diffsptk.ramp(4)
        >>> ulaw = diffsptk.MuLawCompression(4)
        >>> y = ulaw(x)
        >>> y
        tensor([0.0000, 3.0084, 3.5028, 3.7934, 4.0000])

        """
        return self._forward(x, *self.values)

    @staticmethod
    def _func(x, *args, **kwargs):
        values = MuLawCompression._precompute(*args, **kwargs)
        return MuLawCompression._forward(x, *values)

    @staticmethod
    def _check(abs_max, mu):
        if abs_max < 0:
            raise ValueError("abs_max must be non-negative.")
        if mu < 1:
            raise ValueError("mu must be greater than or equal to 1.")

    @staticmethod
    def _precompute(abs_max, mu):
        MuLawCompression._check(abs_max, mu)
        return (
            abs_max,
            mu,
            abs_max / math.log1p(mu),
        )

    @staticmethod
    def _forward(x, abs_max, mu, c):
        x_abs = x.abs() / abs_max
        y = c * torch.sign(x) * torch.log1p(mu * x_abs)
        return y
