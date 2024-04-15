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
import torch.nn.functional as F

from ..misc.utils import TWO_PI
from ..misc.utils import UNVOICED_SYMBOL
from .linear_intpl import LinearInterpolation


class ExcitationGeneration(nn.Module):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/excite.html>`_
    for details.

    Parameters
    ----------
    frame_period : int >= 1
        Frame period in samples, :math:`P`.

    voiced_region : ['pulse', 'sinusoidal', 'sawtooth']
        Value on voiced region.

    unvoiced_region : ['gauss', 'zeros']
        Value on unvoiced region.

    """

    def __init__(self, frame_period, voiced_region="pulse", unvoiced_region="gauss"):
        super().__init__()

        assert 1 <= frame_period
        assert voiced_region in ("pulse", "sinusoidal", "sawtooth")
        assert unvoiced_region in ("gauss", "zeros")

        self.frame_period = frame_period
        self.voiced_region = voiced_region
        self.unvoiced_region = unvoiced_region

    def forward(self, p):
        """Generate a simple excitation signal.

        Parameters
        ----------
        p : Tensor [shape=(..., N)]
            Pitch in seconds.

        Returns
        -------
        out : Tensor [shape=(..., NxP)]
            Excitation signal.

        Examples
        --------
        >>> p = torch.tensor([2.0, 3.0])
        >>> excite = diffsptk.ExcitationGeneration(3)
        >>> e = excite(p)
        >>> e
        tensor([1.4142, 0.0000, 1.6330, 0.0000, 0.0000, 1.7321])

        """
        return self._forward(
            p, self.frame_period, self.voiced_region, self.unvoiced_region
        )

    @staticmethod
    def _forward(p, frame_period, voiced_region, unvoiced_region):
        # Make mask represents voiced region.
        base_mask = torch.clip(p, min=0, max=1)
        mask = torch.ne(base_mask, UNVOICED_SYMBOL)
        mask = torch.repeat_interleave(mask, frame_period, dim=-1)

        # Extend right side for interpolation.
        tmp_mask = F.pad(base_mask, (1, 0))
        tmp_mask = torch.eq(tmp_mask[..., 1:] - tmp_mask[..., :-1], -1)
        p[tmp_mask] = torch.roll(p, 1, dims=-1)[tmp_mask]

        # Interpolate pitch.
        if p.dim() != 1:
            p = p.transpose(-2, -1)
        p = LinearInterpolation._func(p, frame_period)
        if p.dim() != 1:
            p = p.transpose(-2, -1)
        p *= mask

        # Compute phase.
        voiced_pos = torch.gt(p, 0)
        q = torch.zeros_like(p)
        q[voiced_pos] = torch.reciprocal(p[voiced_pos])
        s = torch.cumsum(q.double(), dim=-1)
        bias, _ = torch.cummax(s * ~mask, dim=-1)
        phase = (s - bias).to(p.dtype)

        if voiced_region == "pulse":
            r = torch.ceil(phase)
            r = F.pad(r, (1, 0))
            pulse_pos = torch.ge(r[..., 1:] - r[..., :-1], 1)
            e = torch.zeros_like(p)
            e[pulse_pos] = torch.sqrt(p[pulse_pos])
        elif voiced_region == "sinusoidal":
            e = torch.sin(TWO_PI * phase)
        elif voiced_region == "sawtooth":
            e = torch.fmod(phase, 2) - 1
        else:
            raise ValueError(f"voiced_region {voiced_region} is not supported.")

        if unvoiced_region == "gauss":
            e[~mask] = torch.randn(torch.sum(~mask), device=e.device)
        elif unvoiced_region == "zeros":
            pass
        else:
            raise ValueError(f"unvoiced_region {unvoiced_region} is not supported.")

        return e

    _func = _forward
