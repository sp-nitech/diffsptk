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

from ..misc.utils import to


class HilbertTransform(nn.Module):
    """Hilbert transform module.

    Parameters
    ----------
    fft_length : int >= 1
        Number of FFT bins, should be :math:`T`.

    dim : int
        Dimension along which to take the Hilbert transform.

    """

    def __init__(self, fft_length, dim=-1):
        super().__init__()

        self.dim = dim
        self.register_buffer("h", self._precompute(fft_length))

    def forward(self, x):
        """Compute analytic signal using the Hilbert transform.

        Parameters
        ----------
        x : Tensor [shape=(..., T, ...)]
            Input signal.

        Returns
        -------
        out : Tensor [shape=(..., T, ...)]
            Analytic signal, where real part is the input signal and imaginary part is
            the Hilbert transform of the input signal.

        Examples
        --------
        >>> x = diffsptk.nrand(3)
        >>> x
        tensor([ 1.1809, -0.2834, -0.4169,  0.3883])
        >>> hilbert = diffsptk.HilbertTransform(4)
        >>> z = hilbert(x)
        >>> z.real
        tensor([ 1.1809, -0.2834, -0.4169,  0.3883])
        >>> z.imag
        tensor([ 0.3358,  0.7989, -0.3358, -0.7989])

        """
        return self._forward(x, self.h, self.dim)

    @staticmethod
    def _forward(x, h, dim):
        L = len(h)
        target_shape = [1] * x.dim()
        target_shape[dim] = L
        h = h.view(*target_shape)
        X = torch.fft.fft(x, n=L, dim=dim)
        z = torch.fft.ifft(X * h, n=L, dim=dim)
        return z

    @staticmethod
    def _func(x, fft_length, dim):
        if fft_length is None:
            fft_length = x.size(dim)
        h = HilbertTransform._precompute(fft_length, dtype=x.dtype, device=x.device)
        return HilbertTransform._forward(x, h, dim)

    @staticmethod
    def _precompute(fft_length, dtype=None, device=None):
        h = torch.zeros(fft_length, dtype=torch.double, device=device)
        center = (fft_length + 1) // 2
        h[0] = 1
        h[1:center] = 2
        if fft_length % 2 == 0:
            h[center] = 1
        return to(h, dtype=dtype)
