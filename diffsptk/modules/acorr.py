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

from ..typing import Callable, Precomputed
from ..utils.private import check_size, filter_values
from .base import BaseFunctionalModule


class Autocorrelation(BaseFunctionalModule):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/acorr.html>`_
    for details.

    Parameters
    ----------
    frame_length : int > M
        The frame length in samples, :math:`L`.

    acr_order : int >= 0
        The order of the autocorrelation, :math:`M`.

    out_format : ['naive', 'normalized', 'biased', 'unbiased']
        The type of the autocorrelation.

    """

    def __init__(
        self, frame_length: int, acr_order: int, out_format: str | int = "naive"
    ) -> None:
        super().__init__()

        self.in_dim = frame_length

        self.values = self._precompute(**filter_values(locals()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Estimate the autocorrelation of the input waveform.

        Parameters
        ----------
        x : Tensor [shape=(..., L)]
            The framed waveform.

        Returns
        -------
        out : Tensor [shape=(..., M+1)]
            The autocorrelation.

        Examples
        --------
        >>> import diffsptk
        >>> acorr = diffsptk.Autocorrelation(5, 3)
        >>> x = diffsptk.ramp(4)
        >>> r = acorr(x)
        >>> r
        tensor([30.0000, 20.0000, 11.0000,  4.0000])

        """
        check_size(x.size(-1), self.in_dim, "length of waveform")
        return self._forward(x, *self.values)

    @staticmethod
    def _func(x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        values = Autocorrelation._precompute(x.size(-1), *args, **kwargs)
        return Autocorrelation._forward(x, *values)

    @staticmethod
    def _takes_input_size() -> bool:
        return True

    @staticmethod
    def _check(frame_length: int, acr_order: int) -> None:
        if frame_length <= 0:
            raise ValueError("frame_length must be positive.")
        if frame_length <= acr_order:
            raise ValueError("acr_order must be less than frame_length.")

    @staticmethod
    def _precompute(
        frame_length: int, acr_order: int, out_format: str | int = "naive"
    ) -> Precomputed:
        Autocorrelation._check(frame_length, acr_order)
        if out_format in (0, "naive"):
            formatter = lambda x: x
        elif out_format in (1, "normalized"):
            formatter = lambda x: x / x[..., :1]
        elif out_format in (2, "biased"):
            formatter = lambda x: x / frame_length
        elif out_format in (3, "unbiased"):
            formatter = lambda x: x / (
                torch.arange(
                    frame_length, frame_length - acr_order - 1, -1, device=x.device
                )
            )
        else:
            raise ValueError(f"out_format {out_format} is not supported.")
        return (acr_order, formatter)

    @staticmethod
    def _forward(x: torch.Tensor, acr_order: int, formatter: Callable) -> torch.Tensor:
        fft_length = x.size(-1) + acr_order
        if fft_length % 2 == 1:
            fft_length += 1
        X = torch.fft.rfft(x, n=fft_length).abs().square()
        r = torch.fft.irfft(X)[..., : acr_order + 1]
        r = formatter(r)
        return r
