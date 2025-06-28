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
import torch
import torch.nn.functional as F
from torch import nn

from ..utils.private import to
from .base import BaseNonFunctionalModule
from .pqmf import make_filter_banks


class PseudoQuadratureMirrorFilterBankSynthesis(BaseNonFunctionalModule):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/ipqmf.html>`_
    for details.

    Parameters
    ----------
    n_band : int >= 1
        The number of subbands, :math:`K`.

    filter_order : int >= 2
        The order of the filters, :math:`M`.

    alpha : float > 0
        The stopband attenuation in dB.

    learnable : bool
        Whether to make the filter-bank coefficients learnable.

    device : torch.device or None
        The device of this module.

    dtype : torch.dtype or None
        The data type of this module.

    **kwargs : additional keyword arguments
        The parameters to find optimal filter-bank coefficients.

    References
    ----------
    .. [1] T. Q. Nguyen, "Near-perfect-reconstruction pseudo-QMF banks," *IEEE
           Transactions on Signal Processing*, vol. 42, no. 1, pp. 65-76, 1994.

    .. [2] F. Cruz-Roldan et al., "An efficient and simple method for designing
           prototype filters for cosine-modulated filter banks," *IEEE Signal
           Processing Letters*, vol. 9, no. 1, pp. 29-31, 2002.

    """

    def __init__(
        self,
        n_band: int,
        filter_order: int,
        alpha: float = 100,
        learnable: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        **kwargs,
    ) -> None:
        super().__init__()

        # Make filterbanks.
        filters, is_converged = make_filter_banks(
            n_band, filter_order, mode="synthesis", alpha=alpha, **kwargs
        )
        if not is_converged:
            warnings.warn("Failed to find PQMF coefficients.")
        filters = np.expand_dims(filters, 0)
        filters = np.flip(filters, 2).copy()
        filters = to(filters, device=device, dtype=dtype)
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
            nn.ConstantPad1d((delay_left, 0), 0),
            nn.ReplicationPad1d((0, delay_right)),
        )

    def forward(self, y: torch.Tensor, keepdim: bool = True) -> torch.Tensor:
        """Reconstruct waveform from subband waveforms.

        Parameters
        ----------
        y : Tensor [shape=(B, K, T) or (K, T)]
            The subband waveforms.

        keepdim : bool
            If True, the output shape is (B, 1, T) instead of (B, T).

        Returns
        -------
        out : Tensor [shape=(B, 1, T) or (B, T)]
            The reconstructed waveform.

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
        if y.dim() != 3:
            raise ValueError("Input must be 3D tensor.")

        x = F.conv1d(self.pad(y), self.filters)
        if not keepdim:
            x = x.squeeze(1)
        return x
