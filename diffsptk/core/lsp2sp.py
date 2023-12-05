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

from ..misc.utils import check_size
from ..misc.utils import numpy_to_torch

LOG_ZERO = -1.0e10


class LineSpectralPairsToSpectrum(nn.Module):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/mglsp2sp.html>`_
    for details.

    Parameters
    ----------
    lsp_order : int >= 0 [scalar]
        Order of line spectral pairs, :math:`M`.

    fft_length : int >= 1 [scalar]
        Number of FFT bins, :math:`L`.

    alpha : float [-1 < alpha < 1]
        Warping factor, :math:`\\alpha`.

    gamma : float [-1 <= gamma < 0]
        Gamma, :math:`\\gamma`.

    log_gain : bool [scalar]
        If True, assume input gain is in log scale.

    out_format : ['db', 'log-magnitude', 'magnitude', 'power']
        Output format.

    """

    def __init__(
        self,
        lsp_order,
        fft_length,
        alpha=0,
        gamma=-1,
        log_gain=False,
        out_format="power",
    ):
        super(LineSpectralPairsToSpectrum, self).__init__()

        self.lsp_order = lsp_order
        self.log_gain = log_gain

        assert 0 <= self.lsp_order
        assert 1 <= fft_length
        assert abs(alpha) < 1
        assert -1 <= gamma < 0

        omega = np.linspace(0, np.pi, fft_length // 2 + 1)
        warped_omega = omega + 2 * np.arctan(
            alpha * np.sin(omega) / (1 - alpha * np.cos(omega))
        )
        cos_omega = np.cos(warped_omega).reshape(-1, 1)
        self.register_buffer("cos_omega", numpy_to_torch(cos_omega))

        def floor_log(x):
            return np.log(x, where=0 < x, out=np.full_like(x, LOG_ZERO))

        if lsp_order % 2 == 0:
            p = floor_log(np.sin(0.5 * warped_omega))
            q = floor_log(np.cos(0.5 * warped_omega))
        else:
            p = floor_log(np.sin(warped_omega))
            q = np.zeros_like(warped_omega)
        self.register_buffer("p_bias", numpy_to_torch(p))
        self.register_buffer("q_bias", numpy_to_torch(q))

        self.c1 = 0.5 / gamma
        self.c2 = np.log(2) * (lsp_order if lsp_order % 2 == 0 else lsp_order - 1)

        if out_format == 0 or out_format == "db":
            c = 20 / np.log(10)
            self.convert = lambda x: x * c
        elif out_format == 1 or out_format == "log-magnitude":
            self.convert = lambda x: x
        elif out_format == 2 or out_format == "magnitude":
            self.convert = lambda x: torch.exp(x)
        elif out_format == 3 or out_format == "power":
            self.convert = lambda x: torch.exp(2 * x)
        else:
            raise ValueError(f"out_format {out_format} is not supported")

    def forward(self, w):
        """Convert line spectral pairs to spectrum.

        Parameters
        ----------
        w : Tensor [shape=(..., M+1)]
            Line spectral pairs in radians.

        Returns
        -------
        sp : Tensor [shape=(..., L/2+1)]
            Amplitude spectrum.

        Examples
        --------
        >>> x = diffsptk.nrand(4)
        >>> x
        tensor([ 2.1110, -1.4767,  1.2490,  2.4201,  1.5429])
        >>> lpc = diffsptk.LPC(3, 5)
        >>> a = lpc(x)
        >>> lpc2lsp = diffsptk.LinearPredictiveCoefficientsToLineSpectralPairs(3)
        >>> w = lpc2lsp(a)
        >>> lsp2sp = diffsptk.LineSpectralPairsToSpectrum(3, 8)
        >>> sp = lsp2sp(w)
        >>> sp
        tensor([31.3541, 13.7932, 14.7454, 16.9510, 10.4759])

        """
        check_size(w.size(-1), self.lsp_order + 1, "dimension of LSP")

        def floor_log(x):
            return torch.clip(torch.log(x), min=LOG_ZERO)

        K, w = torch.split(w, [1, self.lsp_order], dim=-1)
        if not self.log_gain:
            K = floor_log(K)

        cos_w = torch.cos(w).unsqueeze(-2)
        pq = floor_log(torch.abs(self.cos_omega - cos_w))  # [..., L/2+1, M]
        p = pq[..., 1::2].sum(-1)
        q = pq[..., 0::2].sum(-1)
        r = torch.logsumexp(
            2 * torch.stack([p + self.p_bias, q + self.q_bias], dim=-1), dim=-1
        )
        sp = K + self.c1 * (self.c2 + r)
        sp = self.convert(sp)
        return sp
