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

    voiced_region : ['pulse', 'sinusoidal', 'sawtooth', 'inverted-sawtooth', \
                     'triangle', 'square']
        Value on voiced region.

    unvoiced_region : ['zeros', 'gauss']
        Value on unvoiced region.

    polarity : ['auto', 'unipolar', 'bipolar']
        Polarity.

    init_phase : ['zeros', 'random']
        Initial phase.

    """

    def __init__(
        self,
        frame_period,
        *,
        voiced_region="pulse",
        unvoiced_region="gauss",
        polarity="auto",
        init_phase="zeros",
    ):
        super().__init__()

        assert 1 <= frame_period
        assert voiced_region in (
            "pulse",
            "sinusoidal",
            "sawtooth",
            "inverted-sawtooth",
            "triangle",
            "square",
        )
        assert unvoiced_region in ("zeros", "gauss")
        assert polarity in ("auto", "unipolar", "bipolar")
        assert init_phase in ("zeros", "random")

        self.frame_period = frame_period
        self.voiced_region = voiced_region
        self.unvoiced_region = unvoiced_region
        self.polarity = polarity
        self.init_phase = init_phase

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
            p,
            self.frame_period,
            self.voiced_region,
            self.unvoiced_region,
            self.polarity,
            self.init_phase,
        )

    @staticmethod
    def _forward(p, frame_period, voiced_region, unvoiced_region, polarity, init_phase):
        # Make mask represents voiced region.
        base_mask = torch.clip(p, min=0, max=1)
        mask = torch.ne(base_mask, UNVOICED_SYMBOL)
        mask = torch.repeat_interleave(mask, frame_period, dim=-1)

        # Extend right side for interpolation.
        tmp_mask = F.pad(base_mask, (1, 0))
        tmp_mask = torch.eq(torch.diff(tmp_mask), -1)
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
        if init_phase == "zeros":
            pass
        elif init_phase == "random":
            phase += torch.rand_like(p[..., :1])
        else:
            raise ValueError(f"init_phase {init_phase} is not supported.")

        # Generate excitation signal using phase.
        if polarity == "auto":
            unipolar = voiced_region == "pulse"
        else:
            unipolar = polarity == "unipolar"
        e = torch.zeros_like(p)
        if voiced_region == "pulse":

            def get_pulse_pos(p):
                r = torch.ceil(p)
                r = F.pad(r, (1, 0))
                return torch.ge(torch.diff(r), 1)

            if unipolar:
                pulse_pos = get_pulse_pos(phase)
                e[pulse_pos] = torch.sqrt(p[pulse_pos])
            else:
                pulse_pos1 = get_pulse_pos(phase)
                pulse_pos2 = get_pulse_pos(0.5 * phase)
                e[pulse_pos1] = torch.sqrt(p[pulse_pos1])
                e[pulse_pos1 & ~pulse_pos2] *= -1
        elif voiced_region == "sinusoidal":
            if unipolar:
                e[mask] = 0.5 * (1 - torch.cos(TWO_PI * phase[mask]))
            else:
                e[mask] = torch.sin(TWO_PI * phase[mask])
        elif voiced_region == "sawtooth":
            if unipolar:
                e[mask] = torch.fmod(phase[mask], 1)
            else:
                e[mask] = 2 * torch.fmod(phase[mask], 1) - 1
        elif voiced_region == "inverted-sawtooth":
            if unipolar:
                e[mask] = 1 - torch.fmod(phase[mask], 1)
            else:
                e[mask] = 1 - 2 * torch.fmod(phase[mask], 1)
        elif voiced_region == "triangle":
            if unipolar:
                e[mask] = torch.abs(2 * torch.fmod(phase[mask] + 0.5, 1) - 1)
            else:
                e[mask] = 2 * torch.abs(2 * torch.fmod(phase[mask] + 0.75, 1) - 1) - 1
        elif voiced_region == "square":
            if unipolar:
                e[mask] = torch.le(torch.fmod(phase[mask], 1), 0.5).to(e.dtype)
            else:
                e[mask] = 2 * torch.le(torch.fmod(phase[mask], 1), 0.5).to(e.dtype) - 1
        else:
            raise ValueError(f"voiced_region {voiced_region} is not supported.")

        if unvoiced_region == "zeros":
            pass
        elif unvoiced_region == "gauss":
            e[~mask] = torch.randn(torch.sum(~mask), device=e.device)
        else:
            raise ValueError(f"unvoiced_region {unvoiced_region} is not supported.")
        return e

    _func = _forward
