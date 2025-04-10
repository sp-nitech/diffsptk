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

from ..typing import Precomputed
from ..utils.private import check_size, get_values, to
from .base import BaseFunctionalModule


class RealValuedInverseFastFourierTransform(BaseFunctionalModule):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/ifft.html>`_
    for details.

    Parameters
    ----------
    fft_length : int >= 2
        The FFT length, :math:`L`.

    out_length : int >= 1 or None
        The output length, :math:`N`.

    learnable : bool
        Whether to make the DFT basis learnable.

    """

    def __init__(
        self, fft_length: int, out_length: int | None = None, learnable: bool = False
    ) -> None:
        super().__init__()

        self.in_dim = fft_length // 2 + 1

        self.values, _, tensors = self._precompute(*get_values(locals()))
        if learnable is True:
            self.W = nn.Parameter(tensors[0])
        elif learnable == "debug":
            self.register_buffer("W", tensors[0])

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """Compute inverse FFT of a complex spectrum.

        Parameters
        ----------
        y : Tensor [shape=(..., L)]
            The complex input spectrum.

        Returns
        -------
        out : Tensor [shape=(..., N)]
            The real output signal.

        Examples
        --------
        >>> x = diffsptk.ramp(1, 3)
        >>> x
        tensor([1., 2., 3.])
        >>> fftr = diffsptk.RealValuedFastFourierTransform(8)
        >>> ifftr = diffsptk.RealValuedInverseFastFourierTransform(8, 3)
        >>> x2 = ifftr(fftr(x))
        >>> x2
        tensor([1., 2., 3.])

        """
        check_size(y.size(-1), self.in_dim, "length of spectrum")
        return self._forward(y, *self.values, **self._buffers, **self._parameters)

    @staticmethod
    def _func(y: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        values, _, _ = RealValuedInverseFastFourierTransform._precompute(
            2 * y.size(-1) - 2, *args, **kwargs
        )
        return RealValuedInverseFastFourierTransform._forward(y, *values)

    @staticmethod
    def _takes_input_size() -> bool:
        return True

    @staticmethod
    def _check(fft_length: int, out_length: int | None) -> None:
        if fft_length <= 0 or fft_length % 2 == 1:
            raise ValueError("fft_length must be positive even.")
        if out_length is not None and (out_length <= 0 or fft_length < out_length):
            raise ValueError("out_length must be in [1, fft_length].")

    @staticmethod
    def _precompute(
        fft_length: int,
        out_length: int | None = None,
        learnable: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Precomputed:
        RealValuedInverseFastFourierTransform._check(fft_length, out_length)

        if out_length is None:
            out_length = fft_length

        if learnable:
            W = torch.fft.ifft(torch.eye(fft_length, device=device, dtype=torch.double))
            W = W[: fft_length // 2 + 1, :out_length]
            W[1:-1] *= 2
            W = torch.cat([W.real, -W.imag], dim=0)
            tensors = (to(W, dtype=dtype),)
        else:
            tensors = None
        return (out_length,), None, tensors

    @staticmethod
    def _forward(
        y: torch.Tensor,
        out_length: int,
        W: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if W is None:
            x = torch.fft.irfft(y)[..., :out_length]
        else:
            y = torch.cat([y.real, y.imag], dim=-1)
            x = torch.matmul(y, W)
        return x
