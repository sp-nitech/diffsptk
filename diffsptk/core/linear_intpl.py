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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..misc.utils import default_dtype


class LinearInterpolation(nn.Module):
    """Perform linear interpolation.

    Note that this is not for lienar_intpl in C the version of SPTK, but for
    filter coefficients interpolation in digital filtering.

    Parameters
    ----------
    upsampling_factor : int >= 1 [scalar]
        Upsampling factor, :math:`P`.

    """

    def __init__(self, upsampling_factor):
        super(LinearInterpolation, self).__init__()

        assert 1 <= upsampling_factor

        # Make upsampling filter.
        w = np.linspace(1, 0, upsampling_factor + 1, dtype=default_dtype())[:-1]
        upsampling_filter = np.stack((w, 1 - w), axis=1)
        upsampling_filter = np.expand_dims(
            upsampling_filter, (1, 3)
        )  # (Out, In, Height, Width)
        self.register_buffer("upsampling_filter", torch.from_numpy(upsampling_filter))

        # Make padding module.
        self.pad = nn.ReplicationPad2d((0, 0, 0, 1))

    def forward(self, x):
        """Interpolate filter coefficients.

        Parameters
        ----------
        x : Tensor [shape=(B, N, D)]
            Filter coefficients.

        Returns
        -------
        y : Tensor [shape=(B, NxP, D)]
            Upsampled filter coefficients.

        Examples
        --------
        >>> x = diffsptk.ramp(2)
        >>> linear_intpl = diffsptk.LinearInterpolation(2)
        >>> y = linear_intpl(x.view(1, -1, 1))
        >>> y.reshape(-1)
        tensor([[0.0000, 0.5000, 1.0000, 1.5000, 2.0000, 2.0000]])

        """
        # Return copy if upsampling factor is one.
        if self.upsampling_filter.size(0) == 1:
            return x

        assert x.dim() == 3, "Input must be 3D tensor"
        B, _, D = x.shape

        x = x.unsqueeze(1)  # (B, 1, N, D)
        x = self.pad(x)

        y = F.conv2d(x, self.upsampling_filter)  # (B, P, N, D)
        y = y.permute(0, 2, 1, 3).reshape(B, -1, D)
        return y
