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
import torch.nn as nn

from ..misc.utils import check_size
from ..misc.utils import numpy_to_torch


class MLSADigitalFilterStabilityCheck(nn.Module):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/mlsacheck.html>`_
    for details.

    Parameters
    ----------
    cep_order : int >= 0 [scalar]
        Order of mel-cepstrum, :math:`M`.

    alpha : float [-1 < alpha < 1]
        Frequency warping factor, :math:`\\alpha`.

    fft_length : int > M [scalar]
        Number of FFT bins, :math:`L`.

    pade_order : [4 <= int <= 7].
        Order of Pade approximation.

    strict : bool [scalar]
        If True, keep maximum log approximation error rather than MLSA filter stability.

    fast : bool [scalar]
        Fast mode.

    threshold : float > 0 [scalar]
        Threshold value. If not given, automatically computed.

    mod_type : ['clip', 'scale']
        Modification type.

    warn_type : ['ignore', 'warn', 'exit']
        Behavior for unstable MLSA.

    """

    def __init__(
        self,
        cep_order,
        alpha=0,
        fft_length=256,
        pade_order=4,
        strict=True,
        fast=True,
        threshold=None,
        mod_type="scale",
        warn_type="warn",
    ):
        super(MLSADigitalFilterStabilityCheck, self).__init__()

        self.cep_order = cep_order
        self.fft_length = fft_length
        self.fast = fast
        self.mod_type = mod_type
        self.warn_type = warn_type

        assert 0 <= self.cep_order
        assert self.mod_type in ("clip", "scale")
        assert self.warn_type in ("ignore", "warn", "exit")
        assert not (self.fast and self.mod_type == "clip")

        if threshold is None:
            if pade_order == 4:
                threshold = 4.5 if strict else 6.20
            elif pade_order == 5:
                threshold = 6.0 if strict else 7.65
            elif pade_order == 6:
                threshold = 7.4 if strict else 9.13
            elif pade_order == 7:
                threshold = 8.9 if strict else 10.6
            else:
                raise ValueError("Unexpected Pade order")
        self.threshold = threshold
        assert 0 < threshold

        alpha_vector = (-alpha) ** np.arange(self.cep_order + 1)
        self.register_buffer("alpha_vector", numpy_to_torch(alpha_vector))

    def forward(self, c1):
        """Check stability of MLSA filter.

        Parameters
        ----------
        c1 : Tensor [shape=(..., M+1)]
            Mel-cepstrum.

        Returns
        -------
        c2 : Tensor [shape=(..., M+1)]
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
        check_size(c1.size(-1), self.cep_order + 1, "dimension of mel-cepstrum")

        gain = (c1 * self.alpha_vector).sum(-1, keepdim=True)

        if self.fast:
            max_amplitude = c1.sum(-1, keepdim=True) - gain
        else:
            c1 = torch.cat((c1[..., :1] - gain, c1[..., 1:]), dim=-1)
            C1 = torch.fft.rfft(c1, n=self.fft_length)
            C1_amplitude = C1.abs()
            max_amplitude, _ = C1_amplitude.max(-1, keepdim=True)
        max_amplitude = torch.clip(max_amplitude, 1e-16)

        if torch.any(self.threshold < max_amplitude):
            if self.warn_type == "ignore":
                pass
            elif self.warn_type == "warn":
                warnings.warn("Unstable MLSA filter")
            elif self.warn_type == "exit":
                raise RuntimeError("Unstable MLSA filter")
            else:
                raise RuntimeError

        if self.mod_type == "clip":
            scale = self.threshold / C1_amplitude
        elif self.mod_type == "scale":
            scale = self.threshold / max_amplitude
        else:
            raise RuntimeError
        scale = torch.clip(scale, max=1)

        if self.fast:
            c0, cX = torch.split(c1, [1, self.cep_order], dim=-1)
            c0 = (c0 - gain) * scale + gain
            cX = cX * scale
            c2 = torch.cat((c0, cX), dim=-1)
        else:
            c2 = torch.fft.irfft(C1 * scale)[..., : self.cep_order + 1]
            c2 = torch.cat((c2[..., :1] + gain, c2[..., 1:]), dim=-1)

        return c2
