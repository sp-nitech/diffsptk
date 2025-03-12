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

import torch.nn.functional as F

from ..utils.private import get_values
from ..utils.private import replicate1
from .base import BaseFunctionalModule


class LinearInterpolation(BaseFunctionalModule):
    """Perform linear interpolation.

    Note that this is not related to `linear_intpl` in the C/C++ version of SPTK, but
    is instead intended for interpolation of filter coefficients in digital filtering.

    Parameters
    ----------
    upsampling_factor : int >= 1
        The upsampling factor, :math:`P`.

    """

    def __init__(self, upsampling_factor):
        super().__init__()

        self.values = self._precompute(*get_values(locals()))

    def forward(self, x):
        """Interpolate filter coefficients.

        Parameters
        ----------
        x : Tensor [shape=(B, N, D) or (N, D) or (N,)]
            The filter coefficients.

        Returns
        -------
        out : Tensor [shape=(B, NxP, D) or (NxP, D) or (NxP,)]
            The upsampled filter coefficients.

        Examples
        --------
        >>> x = diffsptk.ramp(2)
        >>> x
        tensor([0., 1., 2.])
        >>> linear_intpl = diffsptk.LinearInterpolation(2)
        >>> y = linear_intpl(x)
        >>> y
        tensor([0.0000, 0.5000, 1.0000, 1.5000, 2.0000, 2.0000])

        """
        return self._forward(x, *self.values)

    @staticmethod
    def _func(x, *args, **kwargs):
        values = LinearInterpolation._precompute(*args, **kwargs)
        return LinearInterpolation._forward(x, *values)

    @staticmethod
    def _takes_input_size():
        return False

    @staticmethod
    def _check(upsampling_factor):
        if upsampling_factor <= 0:
            raise ValueError("The upsampling factor must be positive.")

    @staticmethod
    def _precompute(upsampling_factor):
        LinearInterpolation._check(upsampling_factor)
        return (upsampling_factor,)

    @staticmethod
    def _forward(x, upsampling_factor):
        if upsampling_factor == 1:
            return x

        d = x.dim()
        if d == 1:
            x = x.view(1, -1, 1)
        elif d == 2:
            x = x.unsqueeze(0)
        if x.dim() != 3:
            raise ValueError("Input must be 1D, 2D, or 3D tensor.")

        B, T, D = x.shape
        x = x.transpose(-2, -1).contiguous()  # (B, D, T)
        x = replicate1(x, left=False)
        x = F.interpolate(
            x,
            size=T * upsampling_factor + 1,
            mode="linear",
            align_corners=True,
        )[..., :-1]  # Remove the padded value.
        y = x.transpose(-2, -1).reshape(B, -1, D)

        if d == 1:
            y = y.view(-1)
        elif d == 2:
            y = y.squeeze(0)
        return y
