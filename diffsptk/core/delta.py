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

import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..misc.utils import default_dtype


class Delta(nn.Module):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/delta.html>`_
    for details.

    Parameters
    ----------
    coefficients : list[float] or list[list[float]]
        Delta coefficients.

    regression_width : int or [int, int]
        Width of 1st (and 2nd) regression coefficients

    strip_static : bool [scalar]
        If true, output only delta components.

    """

    def __init__(self, coefficients=None, regression_width=None, strip_static=False):
        super(Delta, self).__init__()

        if coefficients is not None:
            if regression_width is not None:
                warnings.warn("regression_width is given, but not used")

            if not (
                isinstance(coefficients[0], tuple) or isinstance(coefficients[0], list)
            ):
                coefficients = [coefficients]

            if not strip_static:
                coefficients = [[1]] + coefficients

            max_len = max([len(c) for c in coefficients])
            if max_len % 2 == 0:
                max_len += 1

            window = []
            for c in coefficients:
                diff = max_len - len(c)
                if diff % 2 == 0:
                    left_pad = diff // 2
                    right_pad = diff // 2
                else:
                    left_pad = (diff - 1) // 2
                    right_pad = (diff + 1) // 2
                window.append([0] * left_pad + c + [0] * right_pad)
        elif regression_width is not None:
            if not (
                isinstance(regression_width, tuple)
                or isinstance(regression_width, list)
            ):
                regression_width = [regression_width]

            max_len = max(regression_width)
            max_len = max_len * 2 + 1

            window = []
            if not strip_static:
                w = np.zeros(max_len)
                w[max_len // 2] = 1
                window.append(w)

            # Compute 1st order coefficients
            if True:
                n = regression_width[0]
                z = 1 / (n * (n + 1) * (2 * n + 1) / 3)
                j = np.arange(-n, n + 1)
                pad_width = (max_len - (n * 2 + 1)) // 2
                window.append(np.pad(j * z, pad_width))

            # Compute 2nd order coefficients
            if 2 <= len(regression_width):
                n = regression_width[1]
                a0 = 2 * n + 1
                a1 = a0 * n * (n + 1) / 3
                a2 = a1 * (3 * n * n + 3 * n - 1) / 5
                z = 2.0 / (a2 * a0 - a1 * a1)
                j = np.arange(-n, n + 1)
                pad_width = (max_len - (n * 2 + 1)) // 2
                window.append(np.pad((a0 * j * j - a1) * z, pad_width))
        else:
            raise ValueError("coefficients or regression_width must be given")

        window = np.asarray(window, dtype=default_dtype())
        window = np.reshape(window, [-1, 1, max_len, 1])
        self.register_buffer("window", torch.from_numpy(window))

        self.pad = nn.ReplicationPad2d((0, 0, (max_len - 1) // 2, (max_len - 1) // 2))

    def forward(self, x):
        """Compute delta components.

        Parameters
        ----------
        x : Tensor [shape=(B, T, C)]
            Input.

        Returns
        -------
        y : Tensor [shape=(B, T, CxD)]
            Delta components (with input).

        Examples
        --------
        >>> x = diffsptk.ramp(1, 8).view(1, -1, 2)
        >>> x
        tensor([[[1., 2.],
                 [3., 4.],
                 [5., 6.],
                 [7., 8.]]])
        >>> delta = diffsptk.Delta([[-0.5, 0], [0, 0, 0.5]])
        >>> y = delta(x)
        >>> y
        tensor([[[ 1.0000,  2.0000, -0.5000, -1.0000,  1.5000,  2.0000],
                 [ 3.0000,  4.0000, -0.5000, -1.0000,  2.5000,  3.0000],
                 [ 5.0000,  6.0000, -1.5000, -2.0000,  3.5000,  4.0000],
                 [ 7.0000,  8.0000, -2.5000, -3.0000,  3.5000,  4.0000]]])

        """
        assert x.dim() == 3
        B, T, _ = x.shape

        x = x.unsqueeze(1)
        x = self.pad(x)
        y = F.conv2d(x, self.window, padding="valid")  # (B, D, T, C)
        y = y.permute(0, 2, 1, 3)
        y = y.reshape(B, T, -1)
        return y
