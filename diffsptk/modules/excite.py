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

from ..signals import mseq_like
from ..utils.private import TAU, UNVOICED_SYMBOL, filter_values
from .base import BaseFunctionalModule, Precomputed
from .linear_intpl import LinearInterpolation


def generate_pulse(
    pitch: torch.Tensor,
    phase: torch.Tensor,
    bipolar: bool,
) -> torch.Tensor:
    def get_pulse_pos(p):
        return torch.ge(torch.diff(torch.ceil(p)), 1)

    e = torch.zeros_like(pitch)
    pulse_pos = get_pulse_pos(phase)
    e[pulse_pos] = torch.sqrt(pitch[pulse_pos])

    if bipolar:
        pulse_pos_double = get_pulse_pos(0.5 * phase)
        e[pulse_pos & ~pulse_pos_double] *= -1

    return e


def generate_harmonic_pulse(
    pitch: torch.Tensor,
    phase: torch.Tensor,
    bipolar: bool,
) -> torch.Tensor:
    # The number of harmonics is floor(0.5 * fs / f0) = floor(0.5 * 1 / T)
    n_harmonics = torch.floor(0.5 * pitch)

    # The summation of sinusoids can be computed efficiently using the closed-form
    # expression of the Dirichlet kernel.
    theta = TAU * phase[..., :-1]
    half_theta = 0.5 * theta
    if bipolar:
        numer = torch.cos(half_theta) - torch.cos((n_harmonics + 0.5) * theta)
    else:
        numer = -torch.sin(half_theta) + torch.sin((n_harmonics + 0.5) * theta)
    denom = 2 * torch.sin(half_theta)

    # Handle singularities at phase is zero (i.e., harmonics).
    eps = 1e-6
    is_singular = denom.abs() < eps
    safe_denom = torch.where(is_singular, torch.ones_like(denom), denom)
    e = numer / safe_denom
    if bipolar:
        e[is_singular] = 0
    else:
        e[is_singular] = n_harmonics[is_singular]

    # Normalize as the same energy as the pulse excitation.
    norm_factor = torch.sqrt(2 / n_harmonics.clip(min=1))
    return norm_factor * e


def generate_sinusoidal(phase: torch.Tensor, bipolar: bool) -> torch.Tensor:
    if bipolar:
        e = torch.sin(TAU * phase)
    else:
        e = 0.5 * (1 - torch.cos(TAU * phase))
    return e


def generate_sawtooth(phase: torch.Tensor, bipolar: bool) -> torch.Tensor:
    e = torch.fmod(phase, 1)
    if bipolar:
        e = 2 * e - 1
    return e


def generate_inverted_sawtooth(phase: torch.Tensor, bipolar: bool) -> torch.Tensor:
    e = 1 - torch.fmod(phase, 1)
    if bipolar:
        e = 2 * e - 1
    return e


def generate_triangle(phase: torch.Tensor, bipolar: bool) -> torch.Tensor:
    if bipolar:
        e = 2 * torch.abs(2 * torch.fmod(phase + 0.75, 1) - 1) - 1
    else:
        e = torch.abs(2 * torch.fmod(phase + 0.5, 1) - 1)
    return e


def generate_square(phase: torch.Tensor, bipolar: bool) -> torch.Tensor:
    e = torch.le(torch.fmod(phase, 1), 0.5).to(phase.dtype)
    if bipolar:
        e = 2 * e - 1
    return e


def generate_gauss(source: torch.Tensor) -> torch.Tensor:
    return torch.randn_like(source)


def generate_mseq(source: torch.Tensor) -> torch.Tensor:
    return mseq_like(source)


def generate_uniform(source: torch.Tensor) -> torch.Tensor:
    return math.sqrt(12) * torch.rand_like(source)


class ExcitationGeneration(BaseFunctionalModule):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/excite.html>`_
    for details.

    Parameters
    ----------
    frame_period : int >= 1
        The frame period in samples, :math:`P`.

    voiced_region : ['pulse', 'sinusoidal', 'sawtooth', 'inverted-sawtooth', \
                     'triangle', 'square', 'harmonic-pulse']
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

        self._register_precomputed(self._precompute(**filter_values(locals())))

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
        return self._call_forward(p)

    @staticmethod
    def _func(x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        _p = ExcitationGeneration._precompute(*args, **kwargs)
        return ExcitationGeneration._apply_precomputed(_p, p=x)

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
        return Precomputed(
            values={
                "frame_period": frame_period,
                "voiced_region": voiced_region,
                "unvoiced_region": unvoiced_region,
                "polarity": polarity,
                "init_phase": init_phase,
            }
        )

    @staticmethod
    @torch.inference_mode()
    def _forward(
        p: torch.Tensor,
        *,
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

        # Compute initial phase.
        if not isinstance(init_phase, str):
            shift = init_phase / TAU
        elif init_phase == "zeros":
            shift = 0.0
        elif init_phase == "random":
            shift = torch.rand_like(p[..., :1])
        else:
            raise ValueError(f"init_phase {init_phase} is not supported.")

        # Compute phase.
        voiced_pos = torch.gt(p, 0)
        q = torch.zeros_like(p)
        q[voiced_pos] = torch.reciprocal(p[voiced_pos])
        s = torch.cumsum(q.double(), dim=-1)
        bias, _ = torch.cummax(s * ~mask, dim=-1)
        phase = (s - bias).to(p.dtype)

        if polarity == "auto":
            bipolar = voiced_region != "pulse"
        elif polarity in ("unipolar", "bipolar"):
            bipolar = polarity == "bipolar"
        else:
            raise ValueError(f"polarity {polarity} is not supported.")

        if "pulse" in voiced_region:
            generators = {
                "pulse": generate_pulse,
                "harmonic-pulse": generate_harmonic_pulse,
            }
            if voiced_region not in generators:
                raise ValueError(f"voiced_region {voiced_region} is not supported.")
            phase = F.pad(phase, (1, 0))
            phase += shift
            e = generators[voiced_region](p, phase, bipolar)
        else:
            generators = {
                "sinusoidal": generate_sinusoidal,
                "sawtooth": generate_sawtooth,
                "inverted-sawtooth": generate_inverted_sawtooth,
                "triangle": generate_triangle,
                "square": generate_square,
            }
            if voiced_region not in generators:
                raise ValueError(f"voiced_region {voiced_region} is not supported.")
            phase += shift
            e = torch.zeros_like(p)
            e[mask] = generators[voiced_region](phase[mask], bipolar)

        if unvoiced_region == "zeros":
            pass
        else:
            generators = {
                "gauss": generate_gauss,
                "m-sequence": generate_mseq,
                "uniform": generate_uniform,
            }
            if unvoiced_region not in generators:
                raise ValueError(f"unvoiced_region {unvoiced_region} is not supported.")
            e[~mask] = generators[unvoiced_region](e[~mask])

        return e
