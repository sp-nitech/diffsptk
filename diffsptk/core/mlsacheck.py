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
from ..misc.utils import default_dtype
from ..misc.utils import is_in


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

    threshold : float > 0 [scalar]
        Threshold value. If not given, automatically computed.

    mod_type : ['clip', 'scale', 'fast']
        Modification type.

    warn_type : ['ignore', 'warn', 'exit']
        Behavior for unstable MLSA filter.

    """

    def __init__(
        self,
        cep_order,
        alpha,
        fft_length=None,
        pade_order=4,
        strict=True,
        threshold=None,
        mod_type="fast",
        warn_type="warn",
    ):
        super(MLSADigitalFilterStabilityCheck, self).__init__()

        self.cep_order = cep_order
        self.mod_type = mod_type
        self.warn_type = warn_type

        assert 0 <= self.cep_order
        assert is_in(self.mod_type, ["clip", "scale", "fast"])
        assert is_in(self.warn_type, ["ignore", "warn", "exit"])

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
                raise NotImplementedError
        self.threshold = threshold

        alpha_vector = (-alpha) ** np.arange(self.cep_order + 1, dtype=default_dtype())
        self.register_buffer("alpha_vector", torch.from_numpy(alpha_vector))

    def forward(self, c1):
        """Check stability of MLSA filter.

        Parameters
        ----------
        c1 : Tensor [shape=(..., M+1)]
            Mel-cepstrum.

        Returns
        -------
        c2 : Tensor [shape=(..., M+1)]
            Stabilized mel-cepstrum.

        """
        check_size(c1.size(-1), self.cep_order + 1, "dimension of mel-cepstrum")

        gain = (c1 * self.alpha_vector).sum(-1).unsqueeze(-1)

        if self.mod_type == "fast":
            maximum_amplitude = c1.sum(-1).unsqueeze(-1) - gain
        else:
            raise NotImplementedError

        if torch.any(self.threshold < maximum_amplitude):
            if self.warn_type == "ignore":
                pass
            elif self.warn_type == "warn":  # pragma: no cover
                warnings.warn("Unstable MLSA filter is detected")
            elif self.warn_type == "exit":  # pragma: no cover
                raise RuntimeError("Unstable MLSA filter is detected")
            else:
                raise RuntimeError

        if self.mod_type == "fast":
            scale = self.threshold / maximum_amplitude
            scale = (scale - 1) * (self.threshold < maximum_amplitude) + 1
            scale = torch.clip(scale, max=1)
            c0, cX = torch.split(c1, [1, self.cep_order], dim=-1)
            c0 = (c0 - gain) * scale + gain
            cX = cX * scale
            c2 = torch.cat((c0, cX), dim=-1)
        else:
            raise NotImplementedError

        return c2
