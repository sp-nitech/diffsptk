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

from ..typing import Precomputed
from ..utils.private import check_size, get_values, to
from .base import BaseFunctionalModule


class Window(BaseFunctionalModule):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/window.html>`_
    for details.

    Parameters
    ----------
    in_length : int >= 1
        The window length, :math:`L_1`.

    out_length : int >= L1 or None
        The output length, :math:`L_2`. If :math:`L_2 > L_1`, output is zero-padded.
        If None, :math:`L_2 = L_1`.

    window : ['blackman', 'hamming', 'hanning', 'bartlett', 'trapezoidal', \
              'rectangular', 'nuttall']
        The window type.

    norm : ['none', 'power', 'magnitude']
        The normalization type of the window.

    symmetric : bool
        If True, the window is symmetric, otherwise periodic.

    learnable : bool
        Whether to make the window learnable.

    """

    def __init__(
        self,
        in_length: int,
        out_length: int | None = None,
        *,
        window: str | int = "blackman",
        norm: str | int = "power",
        symmetric: bool = True,
        learnable: bool = False,
    ) -> None:
        super().__init__()

        self.in_dim = in_length

        self.values, _, tensors = self._precompute(*get_values(locals(), drop=1))
        if learnable:
            self.window = nn.Parameter(tensors[0])
        else:
            self.register_buffer("window", tensors[0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply a window function to the given waveform.

        Parameters
        ----------
        x : Tensor [shape=(..., L1)]
            The input framed waveform.

        Returns
        -------
        out : Tensor [shape=(..., L2)]
            The windowed waveform.

        Examples
        --------
        >>> x = torch.ones(5)
        >>> window = diffsptk.Window(5, out_length=7, window="hamming", norm="none")
        >>> y = window(x)
        >>> y
        tensor([0.0800, 0.5400, 1.0000, 0.5400, 0.0800, 0.0000, 0.0000])

        """
        check_size(x.size(-1), self.in_dim, "input length")
        return self._forward(x, *self.values, **self._buffers, **self._parameters)

    @staticmethod
    def _func(x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        values, _, tensors = Window._precompute(
            x.size(-1), *args, **kwargs, device=x.device, dtype=x.dtype
        )
        return Window._forward(x, *values, *tensors)

    @staticmethod
    def _takes_input_size() -> bool:
        return True

    @staticmethod
    def _check(in_length: int, out_length: int | None) -> None:
        if in_length <= 0:
            raise ValueError("in_length must be positive.")
        if out_length is not None and out_length <= 0:
            raise ValueError("out_length must be positive.")

    @staticmethod
    def _precompute(
        in_length: int,
        out_length: int | None,
        window: str | int,
        norm: str | int,
        symmetric: bool,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Precomputed:
        Window._check(in_length, out_length)

        L = in_length
        periodic = not symmetric
        params = {"dtype": dtype, "device": device}
        if window in (0, "blackman"):
            w = torch.blackman_window(L, periodic=periodic, **params)
        elif window in (1, "hamming"):
            w = torch.hamming_window(L, periodic=periodic, **params)
        elif window in (2, "hanning"):
            w = torch.hann_window(L, periodic=periodic, **params)
        elif window in (3, "bartlett"):
            w = torch.bartlett_window(L, periodic=periodic, **params)
        elif window in (4, "trapezoidal"):
            w = (2 * torch.bartlett_window(L, periodic=periodic, **params)).clip(max=1)
        elif window in (5, "rectangular"):
            w = torch.ones(L, **params)
        elif window in (6, "nuttall"):
            size = L if periodic else L - 1
            c1 = torch.tensor([0.355768, -0.487396, 0.144232, -0.012604], **params)
            c2 = torch.arange(0, 8, 2, **params) * (torch.pi / size)
            seed = torch.arange(L, **params)
            w = torch.sum(c1 * torch.cos(torch.outer(seed, c2)), dim=1)
        elif window == "sine":
            w = torch.signal.windows.cosine(L, sym=symmetric, **params)
        elif window == "vorbis":
            seed = torch.signal.windows.cosine(L, sym=symmetric, **params)
            w = torch.sin(torch.pi * 0.5 * seed**2)
        elif window == "kbd":
            if periodic:
                raise ValueError("periodic is not supported for kbd window.")
            seed = torch.kaiser_window(L // 2 + 1, periodic=False, **params)
            cumsum = torch.cumsum(seed, dim=0)
            half = torch.sqrt(cumsum[:-1] / cumsum[-1])
            w = torch.cat([half, half.flip(0)])
        else:
            raise ValueError(f"window {window} is not supported.")

        if norm in (0, "none"):
            pass
        elif norm in (1, "power"):
            w /= torch.sqrt(torch.sum(w**2))
        elif norm in (2, "magnitude"):
            w /= torch.sum(w)
        else:
            raise ValueError(f"norm {norm} is not supported.")

        return (out_length,), None, (to(w, dtype=dtype),)

    @staticmethod
    def _forward(
        x: torch.Tensor, out_length: int | None, window: torch.Tensor
    ) -> torch.Tensor:
        y = x * window
        if out_length is not None:
            in_length = x.size(-1)
            y = F.pad(y, (0, out_length - in_length))
        return y
