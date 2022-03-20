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

import math

import torch
import torch.nn as nn


class CepstralDistance(nn.Module):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/cdist.html>`_
    for details.

    Parameters
    ----------
    full : bool [scalar]
        If true, include the constant term in the distance calculation.

    reduction : ['none', 'mean', 'batchmean', 'sum']
        Reduction type.

    """

    def __init__(self, full=False, reduction="mean"):
        super(CepstralDistance, self).__init__()

        self.full = full
        self.reduction = reduction

        if self.full:
            self.const = 10 * math.sqrt(2) / math.log(10)

    def forward(self, c1, c2):
        """Calculate cepstral distance between two inputs.

        Parameters
        ----------
        c1 : Tensor [shape=(..., M+1)]
            Input cepstral coefficients.

        c2 : Tensor [shape=(..., M+1)]
            Target cepstral coefficients.

        Returns
        -------
        dist : Tensor [shape=(...,) or scalar]
            Cepstral distance.

        Examples
        --------
        >>> c1 = torch.randn(2, 3)
        tensor([[ 0.4296,  1.6517, -0.6022],
                [-1.0464, -0.6088, -0.9274]])
        >>> c2 = torch.randn(2, 3)
        tensor([[ 1.6441, -0.6962, -0.2524],
                [ 0.9344,  0.3965,  1.1494]])
        >>> cdist = diffsptk.CepstralDistance()
        >>> dist = cdist(c1,c2)
        >>> dist
        tensor(1.6551)

        """
        dist = torch.sqrt(torch.square(c1[..., 1:] - c2[..., 1:]).sum(-1))
        if self.reduction == "none":
            pass
        elif self.reduction == "sum":
            dist = dist.sum()
        elif self.reduction == "mean":
            dist = dist.mean() / math.sqrt(c1.shape[-1] - 1)
        elif self.reduction == "batchmean":
            dist = dist.mean()
        else:
            raise ValueError("none, sum, mean, or batchmean is expected")

        if self.full:
            dist *= self.const
        return dist
