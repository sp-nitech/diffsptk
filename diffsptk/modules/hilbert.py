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

from ..typing import Precomputed
from ..utils.private import filter_values, to
from .base import BaseFunctionalModule


class HilbertTransform(BaseFunctionalModule):
    """Hilbert transform module.

    Parameters
    ----------
    fft_length : int >= 1
        The number of FFT bins.

    dim : int
        The dimension along which to take the Hilbert transform.

    device : torch.device or None
        The device of this module.

    dtype : torch.dtype or None
        The data type of this module.

    """

    def __init__(
        self,
        fft_length: int,
        dim: int = -1,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        self.values, _, tensors = self._precompute(**filter_values(locals()))
        self.register_buffer("h", tensors[0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the analytic signal using the Hilbert transform.

        Parameters
        ----------
        x : Tensor [shape=(..., T, ...)]
            The input signal.

        Returns
        -------
        out : Tensor [shape=(..., T, ...)]
            The analytic signal, where the real part is the input signal and the
            imaginary part is the Hilbert transform of the input signal.

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
        return self._forward(x, *self.values, **self._buffers)

    @staticmethod
    def _func(x: torch.Tensor, fft_length: int | None, dim: int) -> torch.Tensor:
        values, _, tensors = HilbertTransform._precompute(
            x.size(dim) if fft_length is None else fft_length,
            dim,
            device=x.device,
            dtype=x.dtype,
        )
        return HilbertTransform._forward(x, *values, *tensors)

    @staticmethod
    def _takes_input_size() -> bool:
        return True

    @staticmethod
    def _check(fft_length: int) -> None:
        if fft_length <= 0:
            raise ValueError("fft_length must be positive.")

    @staticmethod
    def _precompute(
        fft_length: int,
        dim: int,
        device: torch.device | None,
        dtype: torch.dtype | None,
    ) -> Precomputed:
        HilbertTransform._check(fft_length)
        h = torch.zeros(fft_length, device=device, dtype=torch.double)
        center = (fft_length + 1) // 2
        h[0] = 1
        h[1:center] = 2
        if fft_length % 2 == 0:
            h[center] = 1
        return (dim,), None, (to(h, dtype=dtype),)

    @staticmethod
    def _forward(x: torch.Tensor, dim: int, h: torch.Tensor) -> torch.Tensor:
        L = len(h)
        target_shape = [1] * x.dim()
        target_shape[dim] = L
        h = h.view(*target_shape)
        X = torch.fft.fft(x, n=L, dim=dim)
        z = torch.fft.ifft(X * h, n=L, dim=dim)
        return z
