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
from torch import nn

from ..misc.utils import check_size
from ..misc.utils import to

LOG_ZERO = -1.0e10


class LineSpectralPairsToSpectrum(nn.Module):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/mglsp2sp.html>`_
    for details.

    Parameters
    ----------
    lsp_order : int >= 0
        Order of line spectral pairs, :math:`M`.

    fft_length : int >= 1
        Number of FFT bins, :math:`L`.

    alpha : float in (-1, 1)
        Warping factor, :math:`\\alpha`.

    gamma : float in [-1, 0)
        Gamma, :math:`\\gamma`.

    log_gain : bool
        If True, assume input gain is in log scale.

    out_format : ['db', 'log-magnitude', 'magnitude', 'power']
        Output format.

    References
    ----------
    .. [1] A. V. Oppenheim et al., "Discrete representation of signals," *Proceedings of
           the IEEE*, vol. 60, no. 6, pp. 681-691, 1972.

    .. [2] N. Sugamura et al., "Speech data compression by LSP speech analysis-synthesis
           technique," *IEICE trans*, vol. J64-A, no. 8, pp. 599-606, 1981.

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
        super().__init__()

        assert 0 <= lsp_order
        assert 1 <= fft_length
        assert abs(alpha) < 1
        assert -1 <= gamma < 0

        self.lsp_order = lsp_order
        self.log_gain = log_gain
        self.formatter = self._formatter(out_format)

        cos_omega, p_bias, q_bias, self.c1, self.c2 = self._precompute(
            lsp_order, fft_length, alpha, gamma
        )
        self.register_buffer("cos_omega", cos_omega)
        self.register_buffer("p_bias", p_bias)
        self.register_buffer("q_bias", q_bias)

    def forward(self, w):
        """Convert line spectral pairs to spectrum.

        Parameters
        ----------
        w : Tensor [shape=(..., M+1)]
            Line spectral pairs in radians.

        Returns
        -------
        out : Tensor [shape=(..., L/2+1)]
            Spectrum.

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
        return self._forward(
            w,
            self.log_gain,
            self.formatter,
            self.cos_omega,
            self.p_bias,
            self.q_bias,
            self.c1,
            self.c2,
        )

    @staticmethod
    def _forward(w, log_gain, formatter, cos_omega, p_bias, q_bias, c1, c2):
        def floor_log(x):
            return torch.clip(torch.log(x), min=LOG_ZERO)

        K, w = torch.split(w, [1, w.size(-1) - 1], dim=-1)
        if not log_gain:
            K = floor_log(K)

        cos_w = torch.cos(w).unsqueeze(-2)
        pq = floor_log(torch.abs(cos_omega - cos_w))  # [..., L/2+1, M]
        p = pq[..., 1::2].sum(-1)
        q = pq[..., 0::2].sum(-1)
        r = torch.logsumexp(2 * torch.stack([p + p_bias, q + q_bias], dim=-1), dim=-1)
        sp = K + c1 * (c2 + r)
        sp = formatter(sp)
        return sp

    @staticmethod
    def _func(w, fft_length, alpha, gamma, log_gain, out_format):
        formatter = LineSpectralPairsToSpectrum._formatter(out_format)
        precomputes = LineSpectralPairsToSpectrum._precompute(
            w.size(-1) - 1, fft_length, alpha, gamma, dtype=w.dtype, device=w.device
        )
        return LineSpectralPairsToSpectrum._forward(
            w, log_gain, formatter, *precomputes
        )

    @staticmethod
    def _precompute(lsp_order, fft_length, alpha, gamma, dtype=None, device=None):
        omega = torch.linspace(
            0, torch.pi, fft_length // 2 + 1, dtype=torch.double, device=device
        )
        warped_omega = omega + 2 * torch.atan(
            alpha * torch.sin(omega) / (1 - alpha * torch.cos(omega))
        )
        cos_omega = torch.cos(warped_omega).view(-1, 1)
        cos_omega = to(cos_omega, dtype=dtype)

        def floor_log(x):
            return torch.nan_to_num(torch.log(x), nan=LOG_ZERO, neginf=LOG_ZERO)

        if lsp_order % 2 == 0:
            p = floor_log(torch.sin(0.5 * warped_omega))
            q = floor_log(torch.cos(0.5 * warped_omega))
        else:
            p = floor_log(torch.sin(warped_omega))
            q = torch.zeros_like(warped_omega)
        p_bias = to(p, dtype=dtype)
        q_bias = to(q, dtype=dtype)

        c1 = 0.5 / gamma
        c2 = np.log(2) * (lsp_order if lsp_order % 2 == 0 else (lsp_order - 1))
        return cos_omega, p_bias, q_bias, c1, c2

    @staticmethod
    def _formatter(out_format):
        if out_format in (0, "db"):
            c = 20 / np.log(10)
            return lambda x: x * c
        elif out_format in (1, "log-magnitude"):
            return lambda x: x
        elif out_format in (2, "magnitude"):
            return lambda x: torch.exp(x)
        elif out_format in (3, "power"):
            return lambda x: torch.exp(2 * x)
        raise ValueError(f"out_format {out_format} is not supported.")
