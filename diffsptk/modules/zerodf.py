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

from ..typing import Callable, Precomputed
from ..utils.private import check_size, filter_values, to
from .base import BaseFunctionalModule
from .linear_intpl import LinearInterpolation


class AllZeroDigitalFilter(BaseFunctionalModule):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/zerodf.html>`_
    for details.

    Parameters
    ----------
    filter_order : int >= 0
        The order of the filter, :math:`M`.

    frame_period : int >= 1
        The frame period in samples, :math:`P`.

    ignore_gain : bool
        If True, perform filtering without the gain.

    zeroth_index : int >= 0
        The index of the zeroth coefficient in the filter coefficients. If 0, the filter
        is assumed to be minimum-phase. If `M`, the filter is assumed to be
        maximum-phase.

    mode : ['direct', 'efficient']
        The implementation mode for time-varying convolution. 'direct' applies
        convolution at the sample level, linearly interpolating the filter coefficients
        to match the length of the input signal. This approach is simple to understand
        but requires substantial memory. 'efficient' instead performs two separate
        convolutions - one with the original filter coefficients and one with the
        shifted coefficients - and interpolates between their outputs. This avoids the
        need to interpolate the filter coefficients themselves, resulting in lower
        memory usage. Both modes are mathematically equivalent, though 'efficient' may
        produce slight numerical differences owing to the different order of operations.

    device : torch.device or None
        The device of this module.

    dtype : torch.dtype or None
        The data type of this module.

    """

    def __init__(
        self,
        filter_order: int,
        frame_period: int,
        ignore_gain: bool = False,
        zeroth_index: int = 0,
        mode: str = "direct",
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        self.in_dim = filter_order + 1

        self.values, _, tensors = self._precompute(**filter_values(locals()))
        if len(tensors) > 0:
            self.register_buffer("ramp", tensors[0])

    def forward(self, x: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Apply an all-zero digital filter.

        Parameters
        ----------
        x : Tensor [shape=(..., T)]
            The excitation signal.

        b : Tensor [shape=(..., T/P, M+1)]
            The filter coefficients.

        Returns
        -------
        out : Tensor [shape=(..., T)]
            The output signal.

        Examples
        --------
        >>> import diffsptk
        >>> zerodf = diffsptk.AllZeroDigitalFilter(0, 1)
        >>> x = diffsptk.step(4)
        >>> b = diffsptk.ramp(4)
        >>> y = zerodf(x, b.view(-1, 1))
        >>> y
        tensor([0., 1., 2., 3., 4.])

        """
        check_size(b.size(-1), self.in_dim, "dimension of impulse response")
        return self._forward(x, b, *self.values, **self._buffers)

    @staticmethod
    def _takes_input_size() -> bool:
        return True

    @staticmethod
    def _func(x: torch.Tensor, b: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        values, _, tensors = AllZeroDigitalFilter._precompute(
            b.size(-1) - 1, *args, **kwargs, device=b.device, dtype=b.dtype
        )
        return AllZeroDigitalFilter._forward(x, b, *values, *tensors)

    @staticmethod
    def _check(
        filter_order: int,
        frame_period: int,
        ignore_gain: bool,
        zeroth_index: int,
    ) -> None:
        if filter_order < 0:
            raise ValueError("filter_order must be non-negative.")
        if frame_period <= 0:
            raise ValueError("frame_period must be positive.")
        if ignore_gain and zeroth_index not in (0, filter_order):
            raise ValueError(
                "zeroth_index must be 0 or filter_order when ignore_gain is True."
            )
        if zeroth_index < 0 or zeroth_index > filter_order:
            raise ValueError("zeroth_index must be in [0, filter_order].")

    @staticmethod
    def _precompute(
        filter_order: int,
        frame_period: int,
        ignore_gain: bool = False,
        zeroth_index: int = 0,
        mode: str = "direct",
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Precomputed:
        AllZeroDigitalFilter._check(
            filter_order, frame_period, ignore_gain, zeroth_index
        )

        padding = (filter_order - zeroth_index, zeroth_index)

        if mode == "direct":
            impl = AllZeroDigitalFilter._forward_direct
            tensors = ()
        elif mode == "efficient":
            impl = AllZeroDigitalFilter._forward_efficient
            ramp = torch.arange(frame_period, device=device) / frame_period
            ramp = ramp.view(1, 1, -1)
            tensors = (to(ramp, dtype=dtype),)
        else:
            raise ValueError("mode must be 'direct' or 'efficient'.")

        return (frame_period, ignore_gain, padding, impl), None, tensors

    @staticmethod
    def _forward(
        x: torch.Tensor,
        b: torch.Tensor,
        frame_period: int,
        ignore_gain: bool,
        padding: tuple[int, int],
        impl: Callable,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        check_size(x.size(-1), b.size(-2) * frame_period, "sequence length")
        return impl(x, b, frame_period, ignore_gain, padding, *args, **kwargs)

    @staticmethod
    def _forward_direct(
        x: torch.Tensor,
        b: torch.Tensor,
        frame_period: int,
        ignore_gain: bool,
        padding: tuple[int, int],
    ) -> torch.Tensor:
        M = b.size(-1) - 1
        x = F.pad(x, padding)
        x = x.unfold(-1, M + 1, 1)
        h = LinearInterpolation._func(b.flip(-1), frame_period)
        if ignore_gain:
            h = h / (h[..., :1] if padding[0] == 0 else h[..., -1:])
        y = (x * h).sum(-1)
        return y

    @staticmethod
    def _forward_efficient(
        x: torch.Tensor,
        b: torch.Tensor,
        frame_period: int,
        ignore_gain: bool,
        padding: tuple[int, int],
        ramp: torch.Tensor,
    ) -> torch.Tensor:
        x_org_shape = x.shape
        x = x.view(-1, x.size(-1))
        b = b.view(-1, b.size(-2), b.size(-1))
        B, N, L = b.size()
        BN = B * N

        b1 = b.flip(-1)
        b2 = F.pad(b1[:, 1:], (0, 0, 0, 1), mode="replicate")
        weight1 = b1.view(BN, 1, L)
        weight2 = b2.view(BN, 1, L)

        x = F.pad(x, padding)
        x = x.unfold(-1, L - 1 + frame_period, frame_period)
        x = x.reshape(1, BN, L - 1 + frame_period)

        y1 = F.conv1d(x, weight1, groups=BN)
        y2 = F.conv1d(x, weight2, groups=BN)
        y = torch.lerp(y1, y2, ramp)
        y = y.view(*x_org_shape)

        if ignore_gain:
            b0 = b1[..., :1] if padding[0] == 0 else b1[..., -1:]
            g = LinearInterpolation._func(b0, frame_period)
            g = g.reshape(*x_org_shape)
            y = y / g
        return y
