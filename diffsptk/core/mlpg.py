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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .delta import make_window


class MaximumLikelihoodParameterGeneration(nn.Module):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/mlpg.html>`_
    for details. Currently, only global unit variance is supported.

    Note that this module may require large memory due to huge matrix multiplication.

    Parameters
    ----------
    size : int >= 1 [scalar]
        Length of input, :math:`T`.

    seed : list[list[float]] or list[int]
        Delta coefficients or width of 1st (and 2nd) regression coefficients.

    """

    def __init__(self, size, seed=[[-0.5, 0, 0.5], [1, -2, 1]]):
        super(MaximumLikelihoodParameterGeneration, self).__init__()

        assert 1 <= size

        # Make window matrix.
        window = make_window(seed)
        H, L = window.shape
        T = size
        W = np.zeros((T * H, T), dtype=window.dtype)
        W[0, 0] = window[0, (L - 1) // 2]
        W[-H, -1] = W[0, 0]
        for t in range(1, T - 1):
            W[H * t : H * t + H, t - 1 : t - 1 + L] = window

        WS = W.T  # Assume unit variance.
        WSW = np.matmul(WS, W)
        WSW = np.linalg.inv(WSW)
        M = np.matmul(WSW, WS)  # (T, TxH)
        self.register_buffer("M", torch.from_numpy(M))

        self.H = H

    def forward(self, mean):
        """Perform MLPG to obtain static sequence.

        Parameters
        ----------
        mean : Tensor [shape=(B, T, DxH)]
            Time-variant mean vectors with delta components.

        Returns
        -------
        c : Tensor [shape=(B, T, D)]
            Static components.

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
        >>> mlpg = diffsptk.MLPG(y.size(1), [[-0.5, 0], [0, 0, 0.5]])
        >>> c = mlpg(y)
        >>> c
        tensor([[[1., 2.],
                 [3., 4.],
                 [5., 6.],
                 [7., 8.]]])

        """
        B, T, _ = mean.shape
        mean = mean.reshape(B, T * self.H, -1)
        c = torch.einsum("bTd,tT->btd", mean, self.M)
        return c


class ConvolutionalMaximumLikelihoodParameterGeneration(nn.Module):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/mlpg.html>`_
    for details. Currently, only global variance is supported.

    Note that this module cannot accurately compute the both edges of static components.

    Parameters
    ----------
    kernel_size : int >= 1 [scalar]
        Base kernel size.

    seed : list[list[float]] or list[int]
        Delta coefficients or width of 1st (and 2nd) regression coefficients.

    References
    ----------
    .. [1] V. Klimkov, A. Moinet, A. Nadolski, and T. Drugman, "Parameter generation
           algorithms for text-to-speech synthesis with recurrent neural networks," 2018
           IEEE Spoken Language Technology Workshop (SLT), 2018, pp. 626-631.

    """

    def __init__(self, kernel_size=39, seed=[[-0.5, 0, 0.5], [1, -2, 1]]):
        super(ConvolutionalMaximumLikelihoodParameterGeneration, self).__init__()

        assert 1 <= kernel_size
        assert kernel_size % 2 == 1

        # Make window matrix.
        window = make_window(seed)
        H, L = window.shape
        T = max(L, kernel_size)
        W = np.zeros((T * H, T), dtype=window.dtype)
        W[:H, : (L + 1) // 2] = window[:, (L - 1) // 2 :]
        W[-H:, -(L + 1) // 2 :] = window[:, : -(L - 1) // 2]
        for t in range(1, T - 1):
            W[H * t : H * t + H, t - 1 : t - 1 + L] = window

        WS = W.T  # Assume unit variance.
        WSW = np.matmul(WS, W)
        WSW = np.linalg.inv(WSW)
        M = np.matmul(WSW, WS)
        M = M[(T - 1) // 2]
        M = np.reshape(M, (1, 1, -1, 1))
        self.register_buffer("M", torch.from_numpy(M))

        # Make padding module.
        self.pad = nn.ConstantPad2d((0, 0, (T - 1) // 2 * H, (T - 1) // 2 * H), 0)

        self.H = H

    def forward(self, mean, trim=0):
        """Perform MLPG to obtain static sequence.

        Parameters
        ----------
        mean : Tensor [shape=(B, T, DxH)]
            Time-variant mean vectors with delta components.

        trim : int >= 0 [scalar]
            Trimming length, :math:`E`.

        Returns
        -------
        c : Tensor [shape=(B, T-2E, D)]
            Static components.

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
        >>> mlpg = diffsptk.ConvolutionalMaximumLikelihoodParameterGeneration(
        >>>     3, [[-0.5, 0], [0, 0, 0.5]])
        >>> c = mlpg(y, trim=1)
        >>> c
        tensor([[[3.0000, 4.0000],
                 [5.0000, 6.0000]]])

        """
        B, T, _ = mean.shape
        x = mean.reshape(B, 1, T * self.H, -1)
        c = F.conv2d(self.pad(x), self.M, stride=(self.H, 1))
        c = c.squeeze(1)
        if 1 <= trim:
            c = c[:, trim:-trim]
        return c
