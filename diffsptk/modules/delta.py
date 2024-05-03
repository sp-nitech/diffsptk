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
import torch.nn.functional as F

from ..misc.utils import to


class Delta(nn.Module):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/delta.html>`_
    for details.

    Parameters
    ----------
    seed : list[list[float]] or list[int]
        Delta coefficients or width(s) of 1st (and 2nd) regression coefficients.

    static_out : bool
        If False, output only delta components.

    """

    def __init__(self, seed=[[-0.5, 0, 0.5], [1, -2, 1]], static_out=True):
        super().__init__()

        self.register_buffer("window", self._precompute(seed, static_out))

    def forward(self, x):
        """Compute delta components.

        Parameters
        ----------
        x : Tensor [shape=(B, T, D) or (T, D)]
            Static components.

        Returns
        -------
        out : Tensor [shape=(B, T, DxH) or (T, DxH)]
            Delta (and static) components.

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
        return self._forward(x, self.window)

    @staticmethod
    def _forward(x, window):
        d = x.dim()
        if d == 2:
            x = x.unsqueeze(0)
        assert x.dim() == 3, "Input must be 3D tensor."

        B, T, _ = x.shape
        W = window.size(-1)
        pad_width = (W - 1) // 2

        x = x.unsqueeze(1)
        x = F.pad(x, (0, 0, pad_width, pad_width), mode="replicate")
        w = window.view(-1, 1, W, 1)
        y = F.conv2d(x, w, padding="valid")  # (B, H, T, D)
        y = y.permute(0, 2, 1, 3)
        y = y.reshape(B, T, -1)

        if d == 2:
            y = y.squeeze(0)
        return y

    @staticmethod
    def _func(x, seed, static_out):
        window = Delta._precompute(seed, static_out, dtype=x.dtype, device=x.device)
        return Delta._forward(x, window)

    @staticmethod
    def _precompute(seed, static_out, dtype=None, device=None):
        if not isinstance(seed, (tuple, list)):
            raise ValueError("seed must be tuple or list.")

        if isinstance(seed[0], (tuple, list)):
            # Make window from delta coefficients.
            if static_out:
                seed = [[1]] + list(seed)

            max_len = max([len(coefficients) for coefficients in seed])
            if max_len % 2 == 0:
                max_len += 1

            window = []
            for coefficients in seed:
                diff = max_len - len(coefficients)
                if diff % 2 == 0:
                    left_pad = diff // 2
                    right_pad = diff // 2
                else:
                    left_pad = (diff - 1) // 2
                    right_pad = (diff + 1) // 2
                w = torch.tensor(
                    [0] * left_pad + coefficients + [0] * right_pad,
                    dtype=torch.double,
                    device=device,
                )
                window.append(w)
        else:
            # Make window from width of regression coefficients.
            assert 1 <= min(seed)
            max_len = max(seed) * 2 + 1

            window = []
            if static_out:
                w = torch.zeros(max_len, dtype=torch.double, device=device)
                w[(max_len - 1) // 2] = 1
                window.append(w)

            # Compute 1st order coefficients.
            if True:
                n = seed[0]
                z = 1 / (n * (n + 1) * (2 * n + 1) / 3)
                j = torch.arange(-n, n + 1, dtype=torch.double, device=device)
                pad_width = (max_len - (n * 2 + 1)) // 2
                window.append(F.pad(j * z, (pad_width, pad_width)))

            # Compute 2nd order coefficients.
            if 2 <= len(seed):
                n = seed[1]
                a0 = 2 * n + 1
                a1 = a0 * n * (n + 1) / 3
                a2 = a1 * (3 * n * n + 3 * n - 1) / 5
                z = 1 / (2 * (a2 * a0 - a1 * a1))
                j = torch.arange(-n, n + 1, dtype=torch.double, device=device)
                pad_width = (max_len - (n * 2 + 1)) // 2
                window.append(F.pad((a0 * j * j - a1) * z, (pad_width, pad_width)))

            if 3 <= len(seed):
                raise ValueError("3rd order regression is not supported.")

        window = torch.stack(window)  # (H, W)
        return to(window, dtype=dtype)
