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


class Delay(nn.Module):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/delay.html>`_
    for details.

    Parameters
    ----------
    start : int
        Start point, :math:`S`. If negative, advance signal.

    keeplen : bool
        If True, output has the same length of input.

    dim : int
        Dimension along which to shift the tensors.

    """

    def __init__(self, start, keeplen=False, dim=-1):
        super().__init__()

        self.start = start
        self.keeplen = keeplen
        self.dim = dim

    def forward(self, x):
        """Delay signal.

        Parameters
        ----------
        x : Tensor [shape=(..., T, ...)]
            Signal.

        Returns
        -------
        out : Tensor [shape=(..., T-S, ...)] or [shape=(..., T, ...)]
            Delayed signal.

        Examples
        --------
        >>> x = diffsptk.ramp(1, 3)
        >>> delay = diffsptk.Delay(2)
        >>> y = delay(x)
        >>> y
        tensor([0., 0., 1., 2., 3.])

        """
        return self._forward(x, self.start, self.keeplen, self.dim)

    @staticmethod
    def _forward(x, start=0, keeplen=False, dim=-1):
        # Generate zeros if needed.
        if 0 < start or keeplen:
            shape = list(x.shape)
            shape[dim] = abs(start)
            zeros = torch.zeros(*shape, dtype=x.dtype, device=x.device)

        # Delay signal.
        if 0 < start:
            y = torch.cat((zeros, x), dim=dim)
            if keeplen:
                y, _ = torch.split(y, [y.size(dim) - start, start], dim=dim)
            return y

        # Advance signal.
        if start < 0:
            _, y = torch.split(x, [-start, x.size(dim) + start], dim=dim)
            if keeplen:
                y = torch.cat((y, zeros), dim=dim)
            return y

        return x

    _func = _forward
