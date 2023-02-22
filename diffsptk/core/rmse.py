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
import torch.nn as nn


class RootMeanSquaredError(nn.Module):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/rmse.html>`_
    for details.

    Parameters
    ----------
    reduction : ['none', 'mean', 'sum']
        Reduction type.

    eps : float >= 0 [scalar]
        A small value to prevent NaN.

    """

    def __init__(self, reduction="mean", eps=1e-8):
        super(RootMeanSquaredError, self).__init__()

        self.reduction = reduction
        self.eps = eps

        assert self.reduction in ("none", "mean", "sum")
        assert 0 <= self.eps

    def forward(self, x, y):
        """Calculate RMSE.

        Parameters
        ----------
        x : Tensor [shape=(...,)]
            Input.

        y : Tensor [shape=(...,)]
            Target.

        Returns
        -------
        e : Tensor [shape=(...,) or scalar]
            RMSE.

        Examples
        --------
        >>> x = diffsptk.nrand(4)
        >>> x
        tensor([-0.5945, -0.2401,  0.8633, -0.6464,  0.4515])
        >>> y = diffsptk.nrand(4)
        >>> y
        tensor([-0.4025,  0.9367,  1.1299,  3.1986, -0.2832])
        >>> rmse = diffsptk.RootMeanSquaredError()
        >>> e = rmse(x, y)
        >>> e
        tensor(1.8340)

        """
        e = torch.sqrt(torch.square(x - y).mean(-1) + self.eps)
        if self.reduction == "none":
            pass
        elif self.reduction == "sum":
            e = e.sum()
        elif self.reduction == "mean":
            e = e.mean()
        else:
            raise RuntimeError

        return e
