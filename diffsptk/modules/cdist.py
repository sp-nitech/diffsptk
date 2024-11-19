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


class CepstralDistance(nn.Module):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/cdist.html>`_
    for details.

    Parameters
    ----------
    full : bool
        If True, include the constant term in the distance calculation.

    reduction : ['none', 'mean', 'batchmean', 'sum']
        Reduction type.

    References
    ----------
    .. [1] R. F. Kubichek, "Mel-cepstral distance measure for objective speech quality
           assessment," *Proceedings of IEEE Pacific Rim Conference on Communications
           Computers and Signal Processing*, vol. 1, pp. 125-128, 1993.

    """

    def __init__(self, full=False, reduction="mean"):
        super().__init__()

        assert reduction in ("none", "mean", "batchmean", "sum")

        self.full = full
        self.reduction = reduction

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
        out : Tensor [shape=(...,) or scalar]
            Cepstral distance.

        Examples
        --------
        >>> c1 = diffsptk.nrand(2, 2)
        tensor([[ 0.4296,  1.6517, -0.6022],
                [-1.0464, -0.6088, -0.9274]])
        >>> c2 = diffsptk.nrand(2, 2)
        tensor([[ 1.6441, -0.6962, -0.2524],
                [ 0.9344,  0.3965,  1.1494]])
        >>> cdist = diffsptk.CepstralDistance()
        >>> distance = cdist(c1, c2)
        >>> distance
        tensor(1.6551)

        """
        return self._forward(c1, c2, self.full, self.reduction)

    @staticmethod
    def _forward(c1, c2, full, reduction):
        distance = torch.linalg.vector_norm(c1[..., 1:] - c2[..., 1:], ord=2, dim=-1)

        if reduction == "none":
            pass
        elif reduction == "sum":
            distance = distance.sum()
        elif reduction == "mean":
            distance = distance.mean() / (c1.size(-1) - 1) ** 0.5
        elif reduction == "batchmean":
            distance = distance.mean()
        else:
            raise ValueError(f"reduction {reduction} is not supported.")

        if full:
            # Multiply by 10 * math.sqrt(2) / math.log(10)
            distance = distance * 6.141851463713754
        return distance

    _func = _forward
