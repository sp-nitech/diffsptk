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

import warnings

import numpy as np
from torch import nn
import torch.nn.functional as F

from ..misc.utils import numpy_to_torch
from .pqmf import make_filter_banks


class InversePseudoQuadratureMirrorFilterBanks(nn.Module):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/ipqmf.html>`_
    for details.

    Parameters
    ----------
    n_band : int >= 1
        Number of subbands, :math:`K`.

    filter_order : int >= 2
        Order of filter, :math:`M`.

    alpha : float > 0
        Stopband attenuation in dB.

    learnable : bool
        Whether to make filter-bank coefficients learnable.

    **kwargs : additional keyword arguments
        Parameters to find optimal filter-bank coefficients.

    """

    def __init__(self, n_band, filter_order, alpha=100, learnable=False, **kwargs):
        super().__init__()

        assert 1 <= n_band
        assert 2 <= filter_order
        assert 0 < alpha

        # Make filterbanks.
        filters, is_converged = make_filter_banks(
            n_band, filter_order, mode="synthesis", alpha=alpha, **kwargs
        )
        if not is_converged:
            warnings.warn("Failed to find PQMF coefficients.")
        filters = np.expand_dims(filters, 0)
        filters = np.flip(filters, 2).copy()
        filters = numpy_to_torch(filters)
        if learnable:
            self.filters = nn.Parameter(filters)
        else:
            self.register_buffer("filters", filters)

        # Make padding module.
        if filter_order % 2 == 0:
            delay_left = filter_order // 2
            delay_right = filter_order // 2
        else:
            delay_left = (filter_order - 1) // 2
            delay_right = (filter_order + 1) // 2

        self.pad = nn.Sequential(
            nn.ConstantPad1d((delay_left, 0), 0), nn.ReplicationPad1d((0, delay_right))
        )

    def forward(self, y, keepdim=True):
        """Reconstruct waveform from subband waveforms.

        Parameters
        ----------
        y : Tensor [shape=(B, K, T) or (K, T)]
            Subband waveforms.

        keepdim : bool
            If True, the output shape is (B, 1, T) instead (B, T).

        Returns
        -------
        out : Tensor [shape=(B, 1, T) or (B, T)]
            Reconstructed waveform.

        Examples
        --------
        >>> x = torch.arange(0, 1, 0.25)
        >>> x
        tensor([0.0000, 0.2500, 0.5000, 0.7500, 1.0000])
        >>> pqmf = diffsptk.PQMF(2, 10)
        >>> ipqmf = diffsptk.IPQMF(2, 10)
        >>> x2 = ipqmf(pmqf(x), keepdim=False)
        >>> x2
        tensor([[[8.1887e-04, 2.4754e-01, 5.0066e-01, 7.4732e-01, 9.9419e-01]]])

        """
        if y.dim() == 2:
            y = y.unsqueeze(0)
        assert y.dim() == 3, "Input must be 3D tensor."

        x = F.conv1d(self.pad(y), self.filters)
        if not keepdim:
            x = x.squeeze(1)
        return x
