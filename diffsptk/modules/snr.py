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


class SignalToNoiseRatio(nn.Module):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/snr.html>`_
    for details.

    Parameters
    ----------
    frame_length : int >= 1 or None
        Frame length, :math:`L`. If given, calculate segmental SNR.

    full : bool
        If True, include the constant term in the SNR calculation.

    reduction : ['none', 'mean', 'sum']
        Reduction type.

    eps : float >= 0
        A small value to prevent NaN.

    """

    def __init__(self, frame_length=None, full=False, reduction="mean", eps=1e-8):
        super().__init__()

        if frame_length is not None:
            assert 1 <= frame_length
        assert reduction in ("none", "mean", "sum")
        assert 0 <= eps

        self.frame_length = frame_length
        self.full = full
        self.reduction = reduction
        self.eps = eps

    def forward(self, s, sn):
        """Calculate SNR.

        Parameters
        ----------
        s : Tensor [shape=(...,)]
            Signal.

        sn : Tensor [shape=(...,)]
            Signal plus noise.

        Returns
        -------
        out : Tensor [shape=(...,) or scalar]
            Signal-to-noise ratio.

        Examples
        --------
        >>> s = diffsptk.nrand(4)
        >>> s
        tensor([-0.5804, -0.8002, -0.0645,  0.6101,  0.4396])
        >>> n = diffsptk.nrand(4) * 0.1
        >>> n
        tensor([ 0.0854,  0.0485, -0.0826,  0.1455,  0.0257])
        >>> snr = diffsptk.SignalToNoiseRatio(full=True)
        >>> y = snr(s, s + n)
        >>> y
        tensor(16.0614)

        """
        return self._forward(
            s, sn, self.frame_length, self.full, self.reduction, self.eps
        )

    @staticmethod
    def _forward(s, sn, frame_length, full, reduction, eps):
        if frame_length is not None:
            s = s.unfold(-1, frame_length, frame_length)
            sn = sn.unfold(-1, frame_length, frame_length)

        s2 = torch.square(s).sum(-1)
        n2 = torch.square(sn - s).sum(-1)
        snr = torch.log10((s2 + eps) / (n2 + eps))

        if frame_length is not None:
            snr = snr.squeeze(-1)

        if reduction == "none":
            pass
        elif reduction == "sum":
            snr = snr.sum()
        elif reduction == "mean":
            snr = snr.mean()
        else:
            raise ValueError(f"reduction {reduction} is not supported.")

        if full:
            snr = 10 * snr
        return snr

    _func = _forward
