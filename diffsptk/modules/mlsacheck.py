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

import torch
from torch import nn

from ..misc.utils import check_size
from ..misc.utils import to


class MLSADigitalFilterStabilityCheck(nn.Module):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/mlsacheck.html>`_
    for details.

    Parameters
    ----------
    cep_order : int >= 0
        Order of mel-cepstrum, :math:`M`.

    alpha : float in (-1, 1)
        Frequency warping factor, :math:`\\alpha`.

    pade_order : int in [4, 7]
        Order of Pade approximation.

    strict : bool
        If True, keep maximum log approximation error rather than MLSA filter stability.

    threshold : float > 0 or None
        Threshold value. If not given, automatically computed.

    fast : bool
        Enable fast mode (do not use FFT).

    n_fft : int > M
        Number of FFT bins, :math:`L`. Used only in non-fast mode.

    warn_type : ['ignore', 'warn', 'exit']
        Warning type.

    mod_type : ['clip', 'scale']
        Modification type.

    References
    ----------
    .. [1] S. Imai et al., "Mel log spectrum approximation (MLSA) filter for speech
           synthesis," *Electronics and Communications in Japan*, vol. 66, no. 2,
           pp. 11-18, 1983.

    """

    def __init__(
        self,
        cep_order,
        *,
        alpha=0,
        pade_order=4,
        strict=True,
        threshold=None,
        fast=True,
        n_fft=256,
        warn_type="warn",
        mod_type="scale",
    ):
        super().__init__()

        assert 0 <= cep_order
        assert warn_type in ("ignore", "warn", "exit")
        assert mod_type in ("clip", "scale")
        assert not (fast and mod_type == "clip")

        self.cep_order = cep_order
        self.fast = fast
        self.n_fft = n_fft
        self.warn_type = warn_type
        self.mod_type = mod_type
        self.threshold = self._threshold(threshold, pade_order, strict)
        alpha_vector = self._precompute(cep_order, alpha)
        self.register_buffer("alpha_vector", alpha_vector)

    def forward(self, c):
        """Check stability of MLSA filter.

        Parameters
        ----------
        c : Tensor [shape=(..., M+1)]
            Mel-cepstrum.

        Returns
        -------
        out : Tensor [shape=(..., M+1)]
            Modified mel-cepstrum.

        Examples
        --------
        >>> c1 = diffsptk.nrand(4, stdv=10)
        >>> c1
        tensor([ 1.8963,  7.6629,  4.4804,  8.0669, -1.2768])
        >>> mlsacheck = diffsptk.MLSADigitalFilterStabilityCheck(4, warn_type="ignore")
        >>> c2 = mlsacheck(c1)
        >>> c2
        tensor([ 1.3336,  1.7537,  1.0254,  1.8462, -0.2922])

        """
        check_size(c.size(-1), self.cep_order + 1, "dimension of mel-cepstrum")
        return self._forward(
            c,
            self.fast,
            self.n_fft,
            self.warn_type,
            self.mod_type,
            self.threshold,
            self.alpha_vector,
        )

    @staticmethod
    def _forward(c, fast, n_fft, warn_type, mod_type, threshold, alpha_vector):
        gain = (c * alpha_vector).sum(-1, keepdim=True)

        if fast:
            max_amplitude = c.sum(-1, keepdim=True) - gain
        else:
            c1 = torch.cat((c[..., :1] - gain, c[..., 1:]), dim=-1)
            C1 = torch.fft.rfft(c1, n=n_fft)
            C1_amplitude = C1.abs()
            max_amplitude = torch.amax(C1_amplitude, dim=-1, keepdim=True)
        max_amplitude = torch.clip(max_amplitude, min=1e-16)

        if torch.any(threshold < max_amplitude):
            if warn_type == "ignore":
                pass
            elif warn_type == "warn":
                warnings.warn("Detected unstable MLSA filter.")
            elif warn_type == "exit":
                raise RuntimeError("Detected unstable MLSA filter.")
            else:
                raise RuntimeError

        if mod_type == "clip":
            scale = threshold / C1_amplitude
        elif mod_type == "scale":
            scale = threshold / max_amplitude
        else:
            raise RuntimeError
        scale = torch.clip(scale, max=1)

        cep_order = c.size(-1) - 1
        if fast:
            c0, c1 = torch.split(c, [1, cep_order], dim=-1)
            c0 = (c0 - gain) * scale + gain
            c1 = c1 * scale
            c2 = torch.cat((c0, c1), dim=-1)
        else:
            c2 = torch.fft.irfft(C1 * scale)[..., : cep_order + 1]
            c2 = torch.cat((c2[..., :1] + gain, c2[..., 1:]), dim=-1)
        return c2

    @staticmethod
    def _func(
        c, alpha, pade_order, strict, threshold, fast, n_fft, warn_type, mod_type
    ):
        threshold = MLSADigitalFilterStabilityCheck._threshold(
            threshold, pade_order, strict
        )
        alpha_vector = MLSADigitalFilterStabilityCheck._precompute(
            c.size(-1) - 1, alpha, dtype=c.dtype, device=c.device
        )
        return MLSADigitalFilterStabilityCheck._forward(
            c, fast, n_fft, warn_type, mod_type, threshold, alpha_vector
        )

    @staticmethod
    def _precompute(cep_order, alpha, dtype=None, device=None):
        alpha_vector = (-alpha) ** torch.arange(
            cep_order + 1, dtype=torch.double, device=device
        )
        return to(alpha_vector, dtype=dtype)

    @staticmethod
    def _threshold(threshold, pade_order, strict):
        if threshold is not None:
            return threshold
        if pade_order == 4:
            return 4.5 if strict else 6.20
        elif pade_order == 5:
            return 6.0 if strict else 7.65
        elif pade_order == 6:
            return 7.4 if strict else 9.13
        elif pade_order == 7:
            return 8.9 if strict else 10.6
        raise ValueError("Unexpected Pade order.")
