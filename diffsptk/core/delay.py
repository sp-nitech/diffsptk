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


class Delay(nn.Module):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/delay.html>`_
    for details.

    Parameters
    ----------
    start : int [scalar]
        Start point, :math:`S`. If negative, advance signal.

    keeplen : bool [scalar]
        If True, output has the same length of input.

    """

    def __init__(self, start, keeplen=False):
        super(Delay, self).__init__()

        self.start = start
        self.keeplen = keeplen

    def forward(self, x, dim=-1):
        """Delay signal.

        Parameters
        ----------
        x : Tensor [shape=(..., T, ...)]
            Signal.

        dim : int [scalar]
            Dimension along which to shift the tensors.

        Returns
        -------
        y : Tensor [shape=(..., T-S, ...)] or [shape=(..., T, ...)]
            Delayed signal.

        Examples
        --------
        >>> x = torch.arange(1, 4)
        >>> delay = diffsptk.Delay(2)
        >>> y = delay(x)
        >>> y
        tensor([0., 0., 1., 2., 3.])
        >>> delay = diffsptk.Delay(2, keeplen=True)
        >>> y = delay(x)
        >>> y
        tensor([0., 0., 1.])

        """
        # Generate zeros if needed.
        if 0 < self.start or self.keeplen:
            shape = list(x.shape)
            shape[dim] = abs(self.start)
            zeros = torch.zeros(*shape, dtype=x.dtype, device=x.device)

        # Delay signal.
        if 0 < self.start:
            y = torch.cat((zeros, x), dim=dim)
            if self.keeplen:
                y, _ = torch.split(y, [y.size(dim) - self.start, self.start], dim=dim)
            return y

        # Advance signal.
        if self.start < 0:
            _, y = torch.split(x, [-self.start, x.size(dim) + self.start], dim=dim)
            if self.keeplen:
                y = torch.cat((y, zeros), dim=dim)
            return y

        return x
