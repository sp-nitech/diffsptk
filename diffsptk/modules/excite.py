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

import math

import torch
import torch.nn.functional as F

from ..signals import mseq
from ..typing import Precomputed
from ..utils.private import TAU, UNVOICED_SYMBOL, filter_values
from .base import BaseFunctionalModule
from .linear_intpl import LinearInterpolation


class ExcitationGeneration(BaseFunctionalModule):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/excite.html>`_
    for details.

    Parameters
    ----------
    frame_period : int >= 1
        The frame period in samples, :math:`P`.

    voiced_region : ['pulse', 'sinusoidal', 'sawtooth', 'inverted-sawtooth', \
                     'triangle', 'square']
        The type of voiced region.

    unvoiced_region : ['zeros', 'gauss', 'm-sequence', 'uniform']
        The type of unvoiced region.

    polarity : ['auto', 'unipolar', 'bipolar']
        The polarity.

    init_phase : ['zeros', 'random'] or float
        The initial phase in radians.

    """

    def __init__(
        self,
        frame_period: int,
        *,
        voiced_region: str = "pulse",
        unvoiced_region: str = "gauss",
        polarity: str = "auto",
        init_phase: str | float = "zeros",
    ) -> None:
        super().__init__()

        self.values = self._precompute(**filter_values(locals()))

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        """Generate a simple excitation signal.

        Parameters
        ----------
        p : Tensor [shape=(..., N)]
            The pitch in seconds.

        Returns
        -------
        out : Tensor [shape=(..., NxP)]
            The excitation signal.

        Examples
        --------
        >>> import diffsptk
        >>> excite = diffsptk.ExcitationGeneration(3)
        >>> p = torch.tensor([2.0, 3.0])
        >>> e = excite(p)
        >>> e
        tensor([1.4142, 0.0000, 1.6330, 0.0000, 0.0000, 1.7321])

        """
        return self._forward(p, *self.values)

    @staticmethod
    def _func(x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        values = ExcitationGeneration._precompute(*args, **kwargs)
        return ExcitationGeneration._forward(x, *values)

    @staticmethod
    def _takes_input_size() -> bool:
        return False

    @staticmethod
    def _check(frame_period: int) -> None:
        if frame_period <= 0:
            raise ValueError("frame_period must be positive.")

    @staticmethod
    def _precompute(
        frame_period: int,
        voiced_region: str,
        unvoiced_region: str,
        polarity: str,
        init_phase: str | float,
    ) -> Precomputed:
        ExcitationGeneration._check(frame_period)
        return (frame_period, voiced_region, unvoiced_region, polarity, init_phase)

    @staticmethod
    @torch.inference_mode()
    def _forward(
        p: torch.Tensor,
        frame_period: int,
        voiced_region: str,
        unvoiced_region: str,
        polarity: str,
        init_phase: str | float,
    ) -> torch.Tensor:
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
        if not isinstance(init_phase, str):
            shift = init_phase / TAU
        elif init_phase == "zeros":
            shift = 0.0
        elif init_phase == "random":
            shift = torch.rand_like(p[..., :1])
        else:
            raise ValueError(f"init_phase {init_phase} is not supported.")
        if isinstance(shift, torch.Tensor) or shift != 0.0:
            phase += shift

        # Generate excitation signal using phase.
        if polarity == "auto":
            unipolar = voiced_region == "pulse"
        elif polarity in ("unipolar", "bipolar"):
            unipolar = polarity == "unipolar"
        else:
            raise ValueError(f"polarity {polarity} is not supported.")
        e = torch.zeros_like(p)
        if voiced_region == "pulse":

            def get_pulse_pos(p):
                r = torch.ceil(p)
                return torch.ge(torch.diff(r), 1)

            if isinstance(shift, float):
                padded_phase = F.pad(phase, (1, 0), value=shift)
            else:
                padded_phase = torch.cat([shift, phase], dim=-1)

            if unipolar:
                pulse_pos = get_pulse_pos(padded_phase)
                e[pulse_pos] = torch.sqrt(p[pulse_pos])
            else:
                pulse_pos1 = get_pulse_pos(padded_phase)
                pulse_pos2 = get_pulse_pos(0.5 * padded_phase)
                e[pulse_pos1] = torch.sqrt(p[pulse_pos1])
                e[pulse_pos1 & ~pulse_pos2] *= -1
        elif voiced_region == "sinusoidal":
            if unipolar:
                e[mask] = 0.5 * (1 - torch.cos(TAU * phase[mask]))
            else:
                e[mask] = torch.sin(TAU * phase[mask])
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
            e[~mask] = torch.randn(torch.sum(~mask), device=e.device, dtype=e.dtype)
        elif unvoiced_region == "m-sequence":
            e[~mask] = mseq(torch.sum(~mask) - 1, device=e.device, dtype=e.dtype)
        elif unvoiced_region == "uniform":
            e[~mask] = math.sqrt(12) * (
                torch.rand(torch.sum(~mask), device=e.device, dtype=e.dtype) - 0.5
            )
        else:
            raise ValueError(f"unvoiced_region {unvoiced_region} is not supported.")
        return e
