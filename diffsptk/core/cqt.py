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

import numpy as np
import torch
import torch.nn as nn

from ..misc.utils import next_power_of_two
from ..misc.utils import numpy_to_torch
from .frame import Frame


class ConstantQTransform(nn.Module):
    """Perform constant-Q transform. The implementation is based on the simple
    matrix multiplication.

    Parameters
    ----------
    frame_peirod : int >= 1 [scalar]
        Frame period in samples.

    sample_rate : int >= 1 [scalar]
        Sample rate in Hz.

    f_min : float > 0 [scalar]
        Minimum center frequency in Hz.

    f_max : float <= sample_rate // 2 [scalar]
        Maximum center frequency in Hz.

    n_bin_per_octave : int >= 1  [scalar]
        number of bins per octave, :math:`B`.

    References
    ----------
    .. [1] J. C. Brown and M. S. Puckette, "An efficient algorithm for the
           calculation of a constant Q transform," *Journal of the Acoustical
           Society of America*, vol. 92, no. 5, pp. 2698-2701, 1992.

    """

    def __init__(
        self, frame_period, sample_rate, f_min=32.7, f_max=None, n_bin_per_octave=12
    ):
        super(ConstantQTransform, self).__init__()

        if f_max is None:
            f_max = sample_rate / 2

        assert 0 < f_min < f_max <= sample_rate / 2
        assert 1 <= n_bin_per_octave

        B = n_bin_per_octave
        Q = 1 / (2 ** (1 / B) - 1)
        K = math.ceil(B * math.log2(f_max / f_min))
        fft_length = next_power_of_two(math.ceil(Q * sample_rate / f_min))

        temporal_kernels = np.zeros((K, fft_length), dtype=complex)
        for k in range(K):
            f_k = f_min * 2 ** (k / B)
            N_k = 2 * round(Q * sample_rate / f_k / 2) + 1
            n = np.arange(-(N_k - 1) // 2, (N_k - 1) // 2 + 1)
            w = np.hamming(N_k) / N_k
            s = fft_length // 2 + n[0]
            temporal_kernels[k, s : s + N_k] = w * np.exp(
                (2 * np.pi * 1j * Q / N_k) * n
            )

        spectral_kernels = np.fft.fft(temporal_kernels, axis=-1) / fft_length
        assert np.all(spectral_kernels.imag == 0)
        self.register_buffer("kernel", numpy_to_torch(spectral_kernels.T))

        self.frame = Frame(fft_length, frame_period)

    def forward(self, x):
        """Apply CQT to signal.

        Parameters
        ----------
        x : Tensor [shape=(..., T)]
            Signal.

        Returns
        -------
        X : Tensor [shape=(..., N, K)]
            CQT complex output, where N is the number of frames and K is CQ-bin.

        Examples
        --------
        >>> x = diffsptk.sin(99)
        >>> cqt = diffsptk.CQT(100, 8000, n_bin_per_octave=1)
        >>> X = cqt(x).abs()
        >>> X
        tensor([[0.1054, 0.1479, 0.1113, 0.0604, 0.0327, 0.0160, 0.0076]])

        """
        x = self.frame(x)
        X = torch.fft.fft(x, dim=-1)
        X = torch.matmul(X, self.kernel)
        return X
