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
import torch.nn.functional as F
from torch import nn

from ..typing import Callable, Precomputed
from ..utils.private import get_values, to
from .base import BaseFunctionalModule


class RealValuedFastFourierTransform(BaseFunctionalModule):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/fftr.html>`_
    for details.

    Parameters
    ----------
    fft_length : int >= 2
        The FFT length, :math:`L`.

    out_format : ['complex', 'real', 'imaginary', 'amplitude', 'power']
        The output format.

    learnable : bool
        Whether to make the DFT basis learnable. If True, the module performs DFT rather
        than FFT.

    """

    def __init__(
        self,
        fft_length: int,
        out_format: str | int = "complex",
        learnable: bool = False,
    ) -> None:
        super().__init__()

        self.values, _, tensors = self._precompute(*get_values(locals()))
        if learnable is True:
            self.W = nn.Parameter(tensors[0])
        elif learnable == "debug":
            self.register_buffer("W", tensors[0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute FFT of a real signal.

        Parameters
        ----------
        x : Tensor [shape=(..., N)]
            The real input signal.

        Returns
        -------
        out : Tensor [shape=(..., L/2+1)]
            The output spectrum.

        Examples
        --------
        >>> x = diffsptk.ramp(1, 3)
        >>> x
        tensor([1., 2., 3.])
        >>> fftr = diffsptk.RealValuedFastFourierTransform(8, out_format="real")
        >>> y = fftr(x)
        >>> y
        tensor([ 6.0000,  2.4142, -2.0000, -0.4142,  2.0000])

        """
        return self._forward(x, *self.values, **self._buffers, **self._parameters)

    @staticmethod
    def _func(x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        values, _, _ = RealValuedFastFourierTransform._precompute(*args, **kwargs)
        return RealValuedFastFourierTransform._forward(x, *values)

    @staticmethod
    def _takes_input_size() -> bool:
        return False

    @staticmethod
    def _check(fft_length: int | None) -> None:
        if fft_length is not None and (fft_length <= 0 or fft_length % 2 == 1):
            raise ValueError("fft_length must be positive even.")

    @staticmethod
    def _precompute(
        fft_length: int | None,
        out_format: str | int,
        learnable: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Precomputed:
        RealValuedFastFourierTransform._check(fft_length)

        if out_format in (0, "complex"):
            formatter = lambda x: x
        elif out_format in (1, "real"):
            formatter = lambda x: x.real
        elif out_format in (2, "imaginary"):
            formatter = lambda x: x.imag
        elif out_format in (3, "amplitude"):
            formatter = lambda x: x.abs()
        elif out_format in (4, "power"):
            formatter = lambda x: x.abs().square()
        else:
            raise ValueError(f"out_format {out_format} is not supported.")

        if learnable:
            W = torch.fft.fft(torch.eye(fft_length, device=device, dtype=torch.double))
            W = W[..., : fft_length // 2 + 1]
            W = torch.cat([W.real, W.imag], dim=-1)
            tensors = (to(W, dtype=dtype),)
        else:
            tensors = None
        return (fft_length, formatter), None, tensors

    @staticmethod
    def _forward(
        x: torch.Tensor,
        fft_length: int | None,
        formatter: Callable,
        W: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if W is None:
            y = torch.fft.rfft(x, n=fft_length)
        else:
            if fft_length is not None and fft_length != x.size(-1):
                x = F.pad(x, (0, fft_length - x.size(-1)))
            y = torch.matmul(x, W)
            y = torch.complex(*torch.tensor_split(y, 2, dim=-1))
        return formatter(y)
