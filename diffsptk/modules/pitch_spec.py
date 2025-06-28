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
import torch.nn.functional as F
from torch import nn

from ..third_party.world import (
    dc_correction,
    get_windowed_waveform,
    interp1,
    linear_smoothing,
)
from ..typing import Callable
from ..utils.private import TAU, iir, next_power_of_two, to
from .base import BaseNonFunctionalModule
from .frame import Frame
from .spec import Spectrum


class PitchAdaptiveSpectralAnalysis(BaseNonFunctionalModule):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/pitch_spec.html>`_
    for details. Note that the gradients do not propagated through F0.

    Parameters
    ----------
    frame_period : int >= 1
        The frame period in samples, :math:`P`.

    sample_rate : int >= 8000
        The sample rate in Hz.

    fft_length : int >= 1024
        The number of FFT bins, :math:`L`.

    algorithm : ['cheap-trick', 'straight']
        The algorithm to estimate spectral envelpe. The STRAIGHT supports only double
        precision.

    out_format : ['db', 'log-magnitude', 'magnitude', 'power']
        The output format.

    default_f0 : float > 0
        The F0 value used when the input F0 is unvoiced.

    eps : float >= 0
        A small value added to the power spectrum. Please increase this value if you
        encounter numerical instability (valid only if **algorithm** is 'cheap-trick').

    relative_floor : float < 0 or None
        The relative floor of the power spectrum in dB. Please set this value if you
        encounter numerical instability (valid only if **algorithm** is 'cheap-trick').

    device : torch.device or None
        The device of this module.

    dtype : torch.dtype or None
        The data type of this module.

    **kwargs : additional keyword arguments
        Additional keyword arguments passed to the algorithm-specific extractor.

    References
    ----------
    .. [1] M. Morise, "CheapTrick, a spectral envelope estimator for high-quality speech
           synthesis", *Speech Communication*, vol. 67, pp. 1-7, 2015.

    .. [2] H. Kawahara et al., "Restructuring speech representations using a
           pitch-adaptive time-frequency smoothing and an instantaneous-frequency-based
           F0 extraction: Possible role of a repetitive structure in sounds", *Speech
           Communication*, vol. 27, no. 3-4, pp. 187-207, 1999.

    """

    def __init__(
        self,
        frame_period: int,
        sample_rate: int,
        fft_length: int,
        algorithm: str = "cheap-trick",
        out_format: str | int = "power",
        **kwargs,
    ) -> None:
        super().__init__()

        if frame_period <= 0:
            raise ValueError("frame_period must be positive.")
        if sample_rate < 8000:
            raise ValueError("sample_rate must be at least 8000 Hz.")
        if fft_length < 1024:
            raise ValueError("fft_length must be at least 1024.")

        if algorithm == "cheap-trick":
            self.extractor = SpectrumExtractionByCheapTrick(
                frame_period, sample_rate, fft_length, **kwargs
            )
        elif algorithm == "straight":
            self.extractor = SpectrumExtractionBySTRAIGHT(
                frame_period, sample_rate, fft_length, **kwargs
            )
        else:
            raise ValueError(f"algorithm {algorithm} is not supported.")

        self.formatter = self._formatter(out_format)

    def forward(self, x: torch.Tensor, f0: torch.Tensor) -> torch.Tensor:
        """Estimate spectral envelope.

        Parameters
        ----------
        x : Tensor [shape=(..., T)]
            The input waveform.

        f0 : Tensor [shape=(..., T/P)]
            The F0 in Hz.

        Returns
        -------
        out : Tensor [shape=(..., T/P, L/2+1)]
            The spectral envelope.

        Examples
        --------
        >>> x = diffsptk.sin(1000, 80)
        >>> pitch = diffsptk.Pitch(160, 8000, out_format="f0")
        >>> f0 = pitch(x)
        >>> f0.shape
        torch.Size([7])
        >>> pitch_spec = diffsptk.PitchAdaptiveSpectralAnalysis(160, 8000, 1024)
        >>> sp = pitch_spec(x, f0)
        >>> sp.shape
        torch.Size([7, 513])

        """
        sp = self.extractor(x, f0)
        sp = self.formatter(sp)
        return sp

    @staticmethod
    def _formatter(out_format: str | int) -> Callable:
        if out_format in (0, "db"):
            return lambda x: x * (10 / np.log(10))
        elif out_format in (1, "log-magnitude"):
            return lambda x: x / 2
        elif out_format in (2, "magnitude"):
            return lambda x: torch.exp(x / 2)
        elif out_format in (3, "power"):
            return lambda x: torch.exp(x)
        raise ValueError(f"out_format {out_format} is not supported.")


# ----------------------------------------------------------------- #
# Copyright (c) 2010  M. Morise                                     #
#                                                                   #
# All rights reserved.                                              #
#                                                                   #
# Redistribution and use in source and binary forms, with or        #
# without modification, are permitted provided that the following   #
# conditions are met:                                               #
#                                                                   #
# - Redistributions of source code must retain the above copyright  #
#   notice, this list of conditions and the following disclaimer.   #
# - Redistributions in binary form must reproduce the above         #
#   copyright notice, this list of conditions and the following     #
#   disclaimer in the documentation and/or other materials provided #
#   with the distribution.                                          #
# - Neither the name of the M. Morise nor the names of its          #
#   contributors may be used to endorse or promote products derived #
#   from this software without specific prior written permission.   #
#                                                                   #
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND            #
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,       #
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF          #
# MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE          #
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS #
# BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,          #
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED   #
# TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,     #
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON #
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,   #
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY    #
# OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE           #
# POSSIBILITY OF SUCH DAMAGE.                                       #
# ----------------------------------------------------------------- #


class SpectrumExtractionByCheapTrick(nn.Module):
    """Spectral envelope estimation based on CheapTrick."""

    def __init__(
        self,
        frame_period: int,
        sample_rate: int,
        fft_length: int,
        *,
        default_f0: float = 500,
        q1: float = -0.15,
        eps: float = 0,
        relative_floor: float | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        self.frame_period = frame_period
        self.sample_rate = sample_rate
        self.fft_length = fft_length

        # GetF0FloorForCheapTrick()
        self.f_min = 3 * sample_rate / (fft_length - 3)
        if default_f0 < self.f_min:
            raise ValueError(f"default_f0 must be at least {self.f_min}.")

        # GetFFTSizeForCheapTrick()
        min_fft_length = 2 ** (
            1 + int(np.log(3 * sample_rate / self.f_min + 1) / np.log(2))
        )
        if fft_length < min_fft_length:
            raise ValueError(f"fft_length must be at least {min_fft_length}.")

        # Set WORLD constants.
        self.q1 = q1
        self.default_f0 = default_f0

        self.spec = Spectrum(
            fft_length,
            eps=eps,
            relative_floor=relative_floor,
            out_format="power",
        )

        self.register_buffer(
            "ramp", torch.arange(fft_length, device=device, dtype=dtype)
        )

    def forward(self, x: torch.Tensor, f0: torch.Tensor) -> torch.Tensor:
        f0 = torch.where(f0 <= self.f_min, self.default_f0, f0).unsqueeze(-1).detach()

        # GetWindowedWaveform()
        waveform = get_windowed_waveform(
            x,
            f0,
            3,
            0,
            self.frame_period,
            self.sample_rate,
            self.fft_length,
            "hanning",
            True,
            1e-12,
            self.ramp,
        )

        # GetPowerSpectrum()
        power_spectrum = self.spec(waveform)

        # DCCorrection()
        power_spectrum = dc_correction(
            power_spectrum, f0, self.sample_rate, self.fft_length, self.ramp
        )

        # LinearSmoothing()
        power_spectrum = linear_smoothing(
            power_spectrum, f0 * (2 / 3), self.sample_rate, self.fft_length, self.ramp
        )

        # AddInfinitesimalNoise()
        power_spectrum += (
            torch.randn_like(power_spectrum).abs() * torch.finfo(x.dtype).eps
        )

        # SmoothingWithRecovery()
        one_sided_length = self.fft_length // 2 + 1
        quefrency = self.ramp[:one_sided_length] / self.sample_rate
        z = f0 * quefrency
        smoothing_lifter = torch.sinc(z)
        compensation_lifter = (1 - 2 * self.q1) + 2 * self.q1 * torch.cos(TAU * z)
        smoothing_lifter[..., 0] = 1
        cepstrum = torch.fft.irfft(torch.log(power_spectrum))[..., :one_sided_length]
        log_power_spectrum = torch.fft.hfft(
            cepstrum * smoothing_lifter * compensation_lifter,
        )[..., :one_sided_length]
        return log_power_spectrum


# ------------------------------------------------------------------------ #
# Copyright 2018 Hideki Kawahara                                           #
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


class SpectrumExtractionBySTRAIGHT(nn.Module):
    """Spectral envelope estimation based on STRAIGHT."""

    def __init__(
        self,
        frame_period: int,
        sample_rate: int,
        fft_length: int,
        *,
        default_f0: float = 160,
        spectral_exponent: float = 0.6,
        compensation_factor: float = 0.2,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        self.frame_period = frame_period
        self.sample_rate = sample_rate
        self.fft_length = fft_length
        self.default_f0 = default_f0

        self.pc = spectral_exponent
        self.mag = compensation_factor

        from scipy import signal

        b1, a1 = signal.butter(6, 70 / sample_rate * 2, btype="highpass")
        b2, a2 = signal.butter(6, 300 / sample_rate * 2, btype="highpass")
        b3, a3 = signal.butter(6, 3000 / sample_rate * 2, btype="highpass")
        self.register_buffer(
            "b", to(np.stack([b1, b2, b3]), device=device, dtype=dtype)
        )
        self.register_buffer(
            "a", to(np.stack([a1, a2, a3]), device=device, dtype=dtype)
        )

        frame_length_in_msec = 80
        frame_length = sample_rate * frame_length_in_msec // 1000
        if fft_length < frame_length:
            raise ValueError(f"fft_length must be at least {frame_length}.")
        self.frame = Frame(frame_length, frame_period, zmean=True)

        self.register_buffer(
            "ramp",
            torch.arange(max(frame_length * 2, fft_length), device=device, dtype=dtype),
        )

        tt = (self.ramp[:frame_length] + (1 - frame_length / 2)) / sample_rate
        self.register_buffer("tt", tt)

        self.fNominal = 40
        eta = 1
        wGaussian = torch.exp(-torch.pi * (tt * self.fNominal / eta) ** 2)
        wSynchronousBartlett = 1 - torch.abs(tt * self.fNominal)
        wPSGSeed = self.fftfilt(
            wSynchronousBartlett[0 < wSynchronousBartlett],
            F.pad(wGaussian, (0, frame_length)),
        )
        maxValue, maxLocation = torch.max(wPSGSeed, dim=-1)
        wPSGSeed = wPSGSeed / maxValue
        tNominal = (self.ramp[: 2 * frame_length] - maxLocation) / sample_rate
        self.register_buffer("wPSGSeed", wPSGSeed)
        self.register_buffer("tNominal", tNominal)

        one_sided_length = fft_length // 2 + 1
        remaining_length = fft_length - one_sided_length
        ttm = (
            torch.cat(
                [
                    self.ramp[:one_sided_length],
                    self.ramp[:remaining_length] - remaining_length,
                ]
            )
            / sample_rate
        )
        ttm[0] = 1e-5 / sample_rate
        self.register_buffer("ttm", ttm)

        lft = torch.sigmoid(
            ((self.ramp[:fft_length] - fft_length // 2).abs() - fft_length / 30) / 2
        )
        self.register_buffer("lft", lft)

        from pylstraight.core.sp import optimumsmoothing as optimum_smoothing

        ovc = optimum_smoothing(eta, self.pc)
        self.register_buffer("ovc", to(ovc, device=device, dtype=dtype))

        ncw = round(2 * sample_rate / 1000)
        h3 = signal.convolve(
            np.hanning(ncw // 2 + 2)[1:-1],
            np.exp(-1400 / sample_rate * np.arange(2 * ncw + 1)),
            mode="full",
        )
        self.register_buffer("h3", to(h3, device=device, dtype=dtype))

        ipwm = 7
        ipl = round(ipwm / (frame_period / sample_rate * 1000))
        ww = np.hanning(ipl * 2 + 3)[1:-1]
        ww /= np.sum(ww)
        self.register_buffer("ww", to(ww, device=device, dtype=dtype))

        hh = np.array(
            [
                [1, 1, 1, 1],
                [0, 1 / 2, 2 / 3, 3 / 4],
                [0, 0, 1 / 3, 2 / 4],
                [0, 0, 0, 1 / 4],
            ]
        )
        bb = np.linalg.solve(hh, ovc)
        cc = np.array([1, 4, 9, 16])
        tt = np.arange(one_sided_length) / sample_rate
        pb2 = (np.pi / eta**2 + np.pi**2 / 3 * np.sum(bb * cc)) * tt**2
        self.register_buffer("pb2", to(pb2, device=device, dtype=dtype))

    @staticmethod
    def fftfilt(b: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        nb = b.size(-1)
        nx = x.size(-1)
        fft_length = next_power_of_two(nb + nx - 1)
        B = torch.fft.fft(b, n=fft_length)
        X = torch.fft.fft(x, n=fft_length)
        y = torch.fft.ifft(X * B)[..., :nx]
        return y.real

    def forward(self, x: torch.Tensor, f0: torch.Tensor) -> torch.Tensor:
        if x.dtype != torch.double or self.a.dtype != torch.double:
            raise ValueError("Only double precision is supported.")

        eps = 1e-8

        xamp = torch.std(x, dim=-1, keepdim=True)
        scaleconst = 2200
        x = torch.where(xamp < eps, x, x * (scaleconst / xamp))
        xh = iir(x, self.b, self.a, batching=False)
        tx = self.frame(xh[..., 0, :])

        f0 = f0.unsqueeze(-1).detach()
        f0raw = f0
        unvoiced = f0 == 0
        f0 = torch.where(unvoiced, self.default_f0, f0)
        ttf = self.tt * f0

        def safe_div(x, y):
            return x / (y + eps)

        wxe = interp1(
            self.tNominal, self.wPSGSeed, ttf / self.fNominal, method="*linear"
        )
        wxe = safe_div(wxe, torch.linalg.vector_norm(wxe, dim=-1, keepdim=True))
        bcf = 0.36
        wxd = bcf * wxe * torch.sin(torch.pi * ttf)

        one_sided_length = self.fft_length // 2 + 1
        pw = (
            torch.fft.rfft(tx * wxe, n=self.fft_length).abs() ** 2
            + torch.fft.rfft(tx * wxd, n=self.fft_length).abs() ** 2
        )
        pw = torch.clip(pw, min=eps) ** (self.pc / 2)

        f0pr = f0 * (self.fft_length / self.sample_rate) + 1
        f0p = torch.ceil(f0pr).long()
        f0p2 = torch.floor((f0pr + 1) / 2).long()
        f0pm = f0p.max()
        f0p2m = f0p2.max()
        pwx = self.ramp[:f0pm] + 1
        pwxq = f0pr - self.ramp[:f0p2m]
        tmppw = interp1(
            pwx, pw[..., :f0pm], pwxq, method="linear", batching=(False, True)
        )
        tmppw = F.pad(tmppw, (0, one_sided_length - f0p2m))
        mask = self.ramp[:one_sided_length] < f0p2
        pw = torch.where(mask, tmppw, pw)

        ttmf = self.ttm * f0
        ww2t = torch.sinc(3 * ttmf) ** 2
        spw2 = torch.fft.ihfft(ww2t * torch.fft.hfft(pw) * self.lft).real
        wwt = torch.sinc(ttmf) ** 2
        wwt *= (
            self.ovc[0]
            + self.ovc[1] * 2 * torch.cos(TAU * ttmf)
            + self.ovc[2] * 2 * torch.cos(2 * TAU * ttmf)
        )
        spw = safe_div(
            torch.fft.ihfft(wwt * torch.fft.hfft(safe_div(pw, spw2)) * self.lft).real,
            wwt[..., :1],
        )
        spw = torch.clip(spw, min=-100, max=100)
        n2sgram = spw2 * (
            0.175 * torch.log(2 * torch.cosh(4 / 1.4 * spw) + eps) + 0.5 * spw
        )
        n2sgram = torch.clip(n2sgram, min=eps) ** (2 / self.pc)

        nframe = f0.size(-2)
        pwcs = self.fftfilt(
            self.h3, F.pad(xh[..., 1:, :].abs() ** 2, (0, 4 * len(self.h3)))
        )
        end = self.frame_period * nframe
        pwcs = pwcs[..., : end : self.frame_period]
        lbb = round(300 / self.sample_rate * self.fft_length) - 1
        numer = torch.cat(
            [
                torch.sum(n2sgram[..., lbb:], dim=(-1, -2), keepdim=True),
                torch.sum(n2sgram, dim=(-1, -2), keepdim=True),
            ],
            dim=-2,
        )
        denom = torch.sum(pwcs, dim=-1, keepdim=True)
        pwcs = pwcs * safe_div(numer, denom)
        pwch = pwcs[..., 1, :]

        apwt = self.fftfilt(self.ww, F.pad(pwch, (0, len(self.ww))))
        begin = len(self.ww) // 2
        apwt = apwt[..., begin : begin + nframe]
        mmaa = torch.amax(apwt, dim=-1, keepdim=True)
        apwt = torch.where(apwt <= 0, mmaa, apwt)

        dpwt = self.fftfilt(self.ww, F.pad(torch.diff(pwch) ** 2, (0, len(self.ww))))
        dpwt = dpwt[..., begin : begin + nframe]
        dpwt = torch.sqrt(dpwt + eps)
        rr = safe_div(dpwt, apwt)
        lmbd = torch.sigmoid((torch.sqrt(rr) - 0.75) * 20)

        pwc = lmbd * safe_div(pwcs[..., 0, :], torch.sum(n2sgram, dim=-1)) + (1 - lmbd)
        n2sgram = torch.where(unvoiced, n2sgram * pwc.unsqueeze(-1), n2sgram)
        n2sgram = torch.sqrt(torch.abs(n2sgram + eps))

        if 0 < self.mag:
            ccs2 = torch.fft.hfft(n2sgram)[..., :one_sided_length] * torch.clip(
                1 + self.mag * self.pb2 * f0raw**2, max=20
            )
            n2sgram3 = torch.fft.hfft(ccs2, norm="forward")[..., :one_sided_length]
            n2sgram = (n2sgram3.abs() + n2sgram3) / 2 + 0.1

        xamp = xamp.unsqueeze(-1)
        n3sgram = torch.where(xamp < eps, n2sgram, n2sgram * (xamp / scaleconst))
        n3sgram = 2 * torch.log(torch.abs(n3sgram + eps))
        return n3sgram
