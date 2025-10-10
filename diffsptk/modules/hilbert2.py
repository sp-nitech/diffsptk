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

from ..typing import ArrayLike, Precomputed
from ..utils.private import filter_values, to
from .base import BaseFunctionalModule
from .hilbert import HilbertTransform


class TwoDimensionalHilbertTransform(BaseFunctionalModule):
    """2-D Hilbert transform module.

    Parameters
    ----------
    fft_length : int >= 1 or tuple[int, int]
        The number of FFT bins.

    dim : tuple[int, int]
        The dimension along which to take the Hilbert transform.

    device : torch.device or None
        The device of this module.

    dtype : torch.dtype or None
        The data type of this module.

    """

    def __init__(
        self,
        fft_length: ArrayLike[int] | int,
        dim: ArrayLike[int] = (-2, -1),
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
        x : Tensor [shape=(..., T1, T2, ...)]
            The input signal.

        Returns
        -------
        out : Tensor [shape=(..., T1, T2, ...)]
            The analytic signal, where the real part is the input signal and the
            imaginary part is the Hilbert transform of the input signal.

        Examples
        --------
        >>> import diffsptk
        >>> hilbert2 = diffsptk.TwoDimensionalHilbertTransform((1, 4))
        >>> x = diffsptk.ramp(3).unsqueeze(0)
        >>> z = hilbert2(x)
        >>> z.real
        tensor([[0., 1., 2., 3.]])
        >>> z.imag
        tensor([[ 1., -1., -1.,  1.]])

        """
        return self._forward(x, *self.values, **self._buffers)

    @staticmethod
    def _func(
        x: torch.Tensor, fft_length: ArrayLike[int] | int | None, dim: ArrayLike[int]
    ) -> torch.Tensor:
        values, _, tensors = TwoDimensionalHilbertTransform._precompute(
            (x.size(dim[0]), x.size(dim[1])) if fft_length is None else fft_length,
            dim,
            device=x.device,
            dtype=x.dtype,
        )
        return TwoDimensionalHilbertTransform._forward(x, *values, *tensors)

    @staticmethod
    def _takes_input_size() -> bool:
        return True

    @staticmethod
    def _check(dim: ArrayLike[int]) -> None:
        if len(dim) != 2:
            raise ValueError("dim must have length 2.")

    @staticmethod
    def _precompute(
        fft_length: ArrayLike[int] | int,
        dim: ArrayLike[int],
        device: torch.device | None,
        dtype: torch.dtype | None,
    ) -> Precomputed:
        TwoDimensionalHilbertTransform._check(dim)
        if isinstance(fft_length, int):
            fft_length = (fft_length, fft_length)
        _, _, h1 = HilbertTransform._precompute(
            fft_length[0], None, device=device, dtype=torch.double
        )
        _, _, h2 = HilbertTransform._precompute(
            fft_length[1], None, device=device, dtype=torch.double
        )
        h = h1[0].unsqueeze(1) * h2[0].unsqueeze(0)
        return (dim,), None, (to(h, dtype=dtype),)

    @staticmethod
    def _forward(x: torch.Tensor, dim: ArrayLike[int], h: torch.Tensor) -> torch.Tensor:
        L = h.size(dim[0]), h.size(dim[1])
        target_shape = [1] * x.dim()
        target_shape[dim[0]] = L[0]
        target_shape[dim[1]] = L[1]
        h = h.view(*target_shape)
        X = torch.fft.fft2(x, s=L, dim=dim)
        z = torch.fft.ifft2(X * h, s=L, dim=dim)
        return z
