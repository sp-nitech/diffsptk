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
import torch.nn.functional as F

from ..misc.utils import UNVOICED_SYMBOL
from .linear_intpl import LinearInterpolation


class ExcitationGeneration(nn.Module):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/excite.html>`_
    for details.

    Parameters
    ----------
    frame_period : int >= 1 [scalar]
        Frame period in samples, :math:`P`.

    voiced_region : ['pulse', 'sinusoidal', 'sawtooth']
        Value on voiced region.

    unvoiced_region : ['gauss', 'zeros']
        Value on unvoiced region.

    """

    def __init__(self, frame_period, voiced_region="pulse", unvoiced_region="gauss"):
        super(ExcitationGeneration, self).__init__()

        self.frame_period = frame_period
        self.voiced_region = voiced_region
        self.unvoiced_region = unvoiced_region

        assert 1 <= self.frame_period
        assert self.voiced_region in ("pulse", "sinusoidal", "sawtooth")
        assert self.unvoiced_region in ("gauss", "zeros")

        self.linear_intpl = LinearInterpolation(self.frame_period)

    def forward(self, p):
        """Generate a simple excitation signal.

        Parameters
        ----------
        p : Tensor [shape=(..., N)]
            Pitch in seconds.

        Returns
        -------
        e : Tensor [shape=(..., NxP)]
            Excitation signal.

        Examples
        --------
        >>> p = torch.tensor([2.0, 3.0])
        >>> excite = diffsptk.ExcitationGeneration(3)
        >>> e = excite(p)
        >>> e
        tensor([1.4142, 0.0000, 1.6330, 0.0000, 0.0000, 1.7321])

        """
        # Make mask represents voiced region.
        base_mask = torch.clip(p, min=0, max=1)
        mask = torch.ne(base_mask, UNVOICED_SYMBOL)
        mask = torch.repeat_interleave(mask, self.frame_period, dim=-1)

        # Extend right side for interpolation.
        tmp_mask = F.pad(base_mask, (1, 0))
        tmp_mask = torch.eq(tmp_mask[..., 1:] - tmp_mask[..., :-1], -1)
        p[tmp_mask] = torch.roll(p, 1, dims=-1)[tmp_mask]

        # Interpolate pitch.
        if p.dim() != 1:
            p = p.mT
        p = self.linear_intpl(p)
        if p.dim() != 1:
            p = p.mT
        p *= mask

        # Compute phase.
        voiced_pos = torch.gt(p, 0)
        q = torch.zeros_like(p)
        q[voiced_pos] = torch.reciprocal(p[voiced_pos])
        s = torch.cumsum(q.double(), dim=-1)
        bias, _ = torch.cummax(s * ~mask, dim=-1)
        phase = (s - bias).to(p.dtype)

        if self.voiced_region == "pulse":
            r = torch.ceil(phase)
            r = F.pad(r, (1, 0))
            pulse_pos = torch.ge(r[..., 1:] - r[..., :-1], 1)
            e = torch.zeros_like(p)
            e[pulse_pos] = torch.sqrt(p[pulse_pos])
        elif self.voiced_region == "sinusoidal":
            e = torch.sin((2 * torch.pi) * phase)
        elif self.voiced_region == "sawtooth":
            e = torch.fmod(phase, 2) - 1
        else:
            raise RuntimeError

        if self.unvoiced_region == "gauss":
            e[~mask] = torch.randn(torch.sum(~mask), device=e.device)
        elif self.unvoiced_region == "zeros":
            pass
        else:
            raise RuntimeError

        return e
