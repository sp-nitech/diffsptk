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
from .hilbert import HilbertTransform


class TwoDimensionalHilbertTransform(nn.Module):
    """2-D Hilbert transform module.

    Parameters
    ----------
    fft_length : int or list[int]
        Number of FFT bins.

    dim : list[int]
        Dimensions along which to take the Hilbert transform.

    """

    def __init__(self, fft_length, dim=(-2, -1)):
        super().__init__()

        assert len(dim) == 2

        self.dim = dim
        self.register_buffer("h", self._precompute(fft_length))

    def forward(self, x):
        """Compute analytic signal using the Hilbert transform.

        Parameters
        ----------
        x : Tensor [shape=(..., T1, T2, ...)]
            Input signal.

        Returns
        -------
        out : Tensor [shape=(..., T1, T2, ...)]
            Analytic signal, where real part is the input signal and imaginary part is
            the Hilbert transform of the input signal.

        Examples
        --------
        >>> x = diffsptk.nrand(3)
        >>> x
        tensor([[ 1.1809, -0.2834, -0.4169,  0.3883]])
        >>> hilbert2 = diffsptk.TwoDimensionalHilbertTransform((1, 4))
        >>> z = hilbert2(x)
        >>> z.real
        tensor([[ 1.1809, -0.2834, -0.4169,  0.3883]])
        >>> z.imag
        tensor([[ 0.3358,  0.7989, -0.3358, -0.7989]])

        """
        return self._forward(x, self.h, self.dim)

    @staticmethod
    def _forward(x, h, dim):
        L = h.size(dim[0]), h.size(dim[1])
        target_shape = [1] * x.dim()
        target_shape[dim[0]] = L[0]
        target_shape[dim[1]] = L[1]
        h = h.view(*target_shape)
        X = torch.fft.fft2(x, s=L, dim=dim)
        z = torch.fft.ifft2(X * h, s=L, dim=dim)
        return z

    @staticmethod
    def _func(x, fft_length, dim):
        if fft_length is None:
            fft_length = (x.size(dim[0]), x.size(dim[1]))
        h = TwoDimensionalHilbertTransform._precompute(
            fft_length, dtype=x.dtype, device=x.device
        )
        return TwoDimensionalHilbertTransform._forward(x, h, dim)

    @staticmethod
    def _precompute(fft_length, dtype=None, device=None):
        if isinstance(fft_length, int):
            fft_length = (fft_length, fft_length)
        h1 = HilbertTransform._precompute(
            fft_length[0], dtype=torch.double, device=device
        )
        h2 = HilbertTransform._precompute(
            fft_length[1], dtype=torch.double, device=device
        )
        h = h1.unsqueeze(1) * h2.unsqueeze(0)
        return to(h, dtype=dtype)
