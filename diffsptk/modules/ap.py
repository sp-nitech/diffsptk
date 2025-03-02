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

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ..misc.utils import numpy_to_torch
from ..misc.world import dc_correction
from ..misc.world import get_windowed_waveform
from ..misc.world import linear_smoothing
from .base import BaseNonFunctionalModule
from .spec import Spectrum
from .window import Window


class Aperiodicity(BaseNonFunctionalModule):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/ap.html>`_
    for details. Note that the gradients do not propagated through F0.

    Parameters
    ----------
    frame_period : int >= 1
        The frame period in samples, :math:`P`.

    sample_rate : int >= 8000
        The sample rate in Hz.

    fft_length : int >= 16 or None
        The size of double-sided aperiodicity, :math:`L`. If None, the band aperiodicity
        (uninterpolated aperiodicity) is returned as the output.

    algorithm : ['tandem', 'd4c']
        The algorithm to estimate aperiodicity.

    out_format : ['a', 'p', 'a/p', 'p/a']
        The output format.

    lower_bound : float >= 0
        The lower bound of aperiodicity.

    upper_bound : float <= 1
        The upper bound of aperiodicity.

    References
    ----------
    .. [1] H. Kawahara et al., "Simplification and extension of non-periodic excitation
           source representations for high-quality speech manipulation systems,"
           *Proceedings of Interspeech*, pp. 38-41, 2010.

    .. [2] M. Morise, "D4C, a band-aperiodicity estimator for high-quality speech
           synthesis," *Speech Communication*, vol. 84, pp. 57-65, 2016.

    """

    def __init__(
        self,
        frame_period,
        sample_rate,
        fft_length=None,
        algorithm="tandem",
        out_format="a",
        lower_bound=0.001,
        upper_bound=0.999,
        **kwargs,
    ):
        super().__init__()

        if frame_period <= 0:
            raise ValueError("frame_period must be positive.")
        if sample_rate < 8000:
            raise ValueError("sample_rate must be at least 8000 Hz.")
        if fft_length is not None and fft_length < 16:
            raise ValueError("fft_length must be at least 16.")
        if not 0 <= lower_bound < upper_bound <= 1:
            raise ValueError("Invalid lower_bound and upper_bound.")

        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        if algorithm == "tandem":
            self.extractor = AperiodicityExtractionByTANDEM(
                frame_period, sample_rate, fft_length, **kwargs
            )
        elif algorithm == "d4c":
            self.extractor = AperiodicityExtractionByD4C(
                frame_period, sample_rate, fft_length, **kwargs
            )
        else:
            raise ValueError(f"algorithm {algorithm} is not supported.")

        if out_format in (0, "a"):
            self.convert = lambda x: x
        elif out_format in (1, "p"):
            self.convert = lambda x: 1 - x
        elif out_format in (2, "a/p"):
            self.convert = lambda x: x / (1 - x)
        elif out_format in (3, "p/a"):
            self.convert = lambda x: (1 - x) / x
        else:
            raise ValueError(f"out_format {out_format} is not supported.")

    def forward(self, x, f0):
        """Compute aperiodicity measure.

        Parameters
        ----------
        x : Tensor [shape=(B, T) or (T,)]
            The input waveform.

        f0 : Tensor [shape=(B, T/P) or (T/P,)]
            The F0 in Hz.

        Returns
        -------
        out : Tensor [shape=(B, T/P, L/2+1) or (T/P, L/2+1)]
            The aperiodicity.

        Examples
        --------
        >>> x = diffsptk.sin(1000, 80)
        >>> pitch = diffsptk.Pitch(160, 8000, out_format="f0")
        >>> f0 = pitch(x)
        >>> f0.shape
        torch.Size([7])
        >>> aperiodicity = diffsptk.Aperiodicity(160, 16000, 8)
        >>> ap = aperiodicity(x, f0)
        >>> ap
        tensor([[0.1010, 0.9948, 0.9990, 0.9990, 0.9990],
                [0.0010, 0.8419, 0.3644, 0.5912, 0.9590],
                [0.0010, 0.5316, 0.3091, 0.5430, 0.9540],
                [0.0010, 0.3986, 0.1930, 0.4222, 0.9234],
                [0.0010, 0.3627, 0.1827, 0.4106, 0.9228],
                [0.0010, 0.3699, 0.1827, 0.4106, 0.9228],
                [0.0010, 0.7659, 0.7081, 0.8378, 0.9912]])

        """
        d = x.dim()
        if d == 1:
            x = x.unsqueeze(0)
        if x.dim() != 2:
            raise ValueError("Input must be 1D or 2D tensor.")

        if f0.dim() == 1:
            f0 = f0.unsqueeze(0)
        if f0.dim() != 2:
            raise ValueError("F0 must be 1D or 2D tensor.")

        ap = self.extractor(x, f0)
        ap = torch.clip(ap, min=self.lower_bound, max=self.upper_bound)
        ap = self.convert(ap)

        if d == 1:
            ap = ap.squeeze(0)
        return ap


class AperiodicityExtractionByTANDEM(nn.Module):
    """Aperiodicity extraction by TANDEM-STRAIGHT."""

    def __init__(
        self,
        frame_period,
        sample_rate,
        fft_length=None,
        *,
        window_length_ms=30,
        eps=1e-5,
    ):
        super().__init__()

        if window_length_ms <= 0:
            raise ValueError("window_length_ms must be positive.")
        if eps <= 0:
            raise ValueError("eps must be positive.")

        self.frame_period = frame_period
        self.sample_rate = sample_rate
        self.n_band = int(np.log2(sample_rate / 600))

        self.default_f0 = 150
        self.n_trial = 10

        self.cutoff_list = [sample_rate / 2**i for i in range(2, self.n_band + 1)]
        self.cutoff_list.append(self.cutoff_list[-1])

        if fft_length is not None:
            coarse_axis = [sample_rate / 2**i for i in range(self.n_band, 0, -1)]
            coarse_axis.insert(0, 0)
            coarse_axis = np.asarray(coarse_axis)
            freq_axis = np.arange(fft_length // 2 + 1) * (sample_rate / fft_length)

            idx = np.searchsorted(coarse_axis, freq_axis) - 1
            idx = np.clip(idx, 0, len(coarse_axis) - 2)
            idx = idx.reshape(1, 1, -1)
            self.register_buffer("interp_indices", numpy_to_torch(idx).long())

            x0 = coarse_axis[:-1]
            dx = coarse_axis[1:] - x0
            weights = (freq_axis - np.take(x0, idx)) / np.take(dx, idx)
            self.register_buffer("interp_weights", numpy_to_torch(weights))

        self.segment_length = [
            int(i * window_length_ms / 500 + 1.5) for i in self.cutoff_list
        ]
        ramp = torch.arange(-1, self.segment_length[0] + 1).view(1, 1, -1)
        self.register_buffer("ramp", ramp)
        self.register_buffer("eye", torch.eye(6) * eps)

        hHP = self._qmf_high()
        hLP = self._qmf_low()
        self.register_buffer("hHP", numpy_to_torch(hHP).view(1, 1, -1))
        self.register_buffer("hLP", numpy_to_torch(hLP).view(1, 1, -1))
        self.hHP_pad = nn.ReflectionPad1d(self.hHP.size(-1) // 2)
        self.hLP_pad = nn.ReflectionPad1d(self.hLP.size(-1) // 2)

        window = np.zeros((self.n_band, self.segment_length[0]))
        for i, s in enumerate(self.segment_length):
            window[i, :s] = np.hanning(s + 2)[1:-1]
        self.register_buffer("window", numpy_to_torch(window))
        self.register_buffer("window_sqrt", self.window.sqrt())

    def forward(self, x, f0):
        f0 = torch.where(f0 <= 32, self.default_f0, f0).detach()

        B, N = f0.shape
        time_axis = torch.arange(N, dtype=f0.dtype, device=f0.device) * (
            self.frame_period / self.sample_rate
        )

        bap = []
        lx = x.unsqueeze(1)
        for i in range(self.n_band):
            if i < self.n_band - 1:
                hx = F.conv1d(self.hHP_pad(lx), self.hHP, stride=2)
                lx = F.conv1d(self.hLP_pad(lx), self.hLP, stride=2)
                x = hx
            else:
                x = lx

            tmp_fs = 2 * self.cutoff_list[i]
            pitch = tmp_fs / f0
            t0 = (pitch + 0.5).int()
            index_bias = (pitch * 0.5 + 0.5).int()
            curr_pos = (time_axis * tmp_fs + 1.5).int().unsqueeze(0)  # (1, N)
            origin = curr_pos - index_bias  # (B, N)

            j = self.ramp[..., : self.segment_length[i] + 2]
            xx = x.expand(-1, N, -1)
            T1 = x.size(-1) - 1

            index_alpha = (origin - t0).unsqueeze(-1) + j  # (B, N, J + 2)
            index_alpha = torch.clip(index_alpha, min=0, max=T1)
            H_alpha = torch.gather(xx, -1, index_alpha)
            H_alpha = H_alpha.unfold(2, 3, 1)  # (B, N, J, 3)

            index_beta = (origin + t0).unsqueeze(-1) + j  # (B, N, J + 2)
            index_beta = torch.clip(index_beta, min=0, max=T1)
            H_beta = torch.gather(xx, -1, index_beta)
            H_beta = H_beta.unfold(2, 3, 1)  # (B, N, J, 3)

            H = torch.cat((H_alpha, H_beta), dim=-1)  # (B, N, J, 6)
            w = self.window[i, : self.segment_length[i]]  # (J,)
            Hw = H.transpose(-2, -1) * w  # (B, N, 6, J)
            R = torch.matmul(Hw, H)  # (B, N, 6, 6)

            index_gamma = origin.unsqueeze(-1) + j[..., 1:-1]  # (B, N, J)
            index_gamma = torch.clip(index_gamma, 0, T1)
            X = torch.gather(xx, -1, index_gamma).unsqueeze(-1)

            for n in range(self.n_trial):
                m = 10**n
                u, info = torch.linalg.cholesky_ex(R + self.eye * m)
                if 0 == info.sum().item():
                    if n == self.n_trial - 1:
                        raise RuntimeError("Failed to compute Cholesky decomposition.")
                    break

            b = torch.matmul(Hw, X)  # (B, N, 6, 1)
            a = torch.cholesky_solve(b, u)
            Ha = torch.matmul(H, a)  # (B, N, J, 1)

            wsqrt = self.window_sqrt[i, : self.segment_length[i]]
            wx = wsqrt * X.squeeze(-1)
            wxHa = wsqrt * (X - Ha).squeeze(-1)
            denom = wx.std(dim=-1, unbiased=True)
            numer = wxHa.std(dim=-1, unbiased=True)
            A = numer / (denom + 1e-16)
            bap.append(A)

        bap.append(bap[-1])
        ap = torch.stack(bap[::-1], dim=-1)  # (B, N, D)

        # Interpolate band aperiodicity.
        if hasattr(self, "interp_indices"):
            y = torch.log(ap)
            y0 = y[..., :-1]
            dy = y[..., 1:] - y0
            index = self.interp_indices.expand(B, N, -1)
            y = torch.gather(dy, -1, index) * self.interp_weights
            y += torch.gather(y0, -1, index)
            ap = torch.exp(y)
        return ap

    def _qmf_high(self, dtype=np.float64):
        hHP = np.zeros(41, dtype=dtype)
        hHP[0] = +0.00041447996898231424
        hHP[1] = +0.00078125051417292477
        hHP[2] = -0.0010917236836275842
        hHP[3] = -0.0019867925675967589
        hHP[4] = +0.0020903896961562292
        hHP[5] = +0.0040940570272849346
        hHP[6] = -0.0034025808529816698
        hHP[7] = -0.0074961541272056016
        hHP[8] = +0.0049722633399330637
        hHP[9] = +0.012738791249119802
        hHP[10] = -0.0066960326895749113
        hHP[11] = -0.020694051570247052
        hHP[12] = +0.0084324365650413451
        hHP[13] = +0.033074383758700532
        hHP[14] = -0.010018936738799522
        hHP[15] = -0.054231361405808247
        hHP[16] = +0.011293988915051487
        hHP[17] = +0.10020081367388213
        hHP[18] = -0.012120546202484579
        hHP[19] = -0.31630021039095702
        hHP[20] = +0.51240682580627639
        hHP[21:] = hHP[19::-1]
        return hHP

    def _qmf_low(self, dtype=np.float64):
        hLP = np.zeros(37, dtype=dtype)
        hLP[0] = -0.00065488170077483048
        hLP[1] = +0.00007561994958159384
        hLP[2] = +0.0020408456937895227
        hLP[3] = -0.00074680535322030437
        hLP[4] = -0.0043502235688264931
        hLP[5] = +0.0025966428382642732
        hLP[6] = +0.0076396022827566962
        hLP[7] = -0.0064904118901497852
        hLP[8] = -0.011765804538954506
        hLP[9] = +0.013649908479276255
        hLP[10] = +0.01636866479016021
        hLP[11] = -0.026075976030529347
        hLP[12] = -0.020910294856659444
        hLP[13] = +0.048260725032316647
        hLP[14] = +0.024767846611048111
        hLP[15] = -0.096178467583360641
        hLP[16] = -0.027359756709866623
        hLP[17] = +0.31488052161630042
        hLP[18] = +0.52827343594055032
        hLP[19:] = hLP[17::-1]
        return hLP


class AperiodicityExtractionByD4C(nn.Module):
    """Aperiodicity extraction by D4C."""

    def __init__(
        self,
        frame_period,
        sample_rate,
        fft_length=None,
        *,
        threshold=0.0,
        default_f0=150,
    ):
        super().__init__()

        self.frame_period = frame_period
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.default_f0 = default_f0

        freqency_interval = 3000
        upper_limit = 15000
        floor_f0 = 47
        self.lowest_f0 = 40

        self.fft_length_love = 2 ** (
            1 + int(np.log(3 * sample_rate / self.lowest_f0 + 1) / np.log(2))
        )
        self.fft_length_d4c = 2 ** (
            1 + int(np.log(4 * sample_rate / floor_f0 + 1) / np.log(2))
        )

        n_aperiodicity = int(
            min(upper_limit, sample_rate / 2 - freqency_interval) / freqency_interval
        )
        window_length = freqency_interval * self.fft_length_d4c // sample_rate * 2 + 1
        half_window_length = window_length // 2
        padded_window_length = self.fft_length_d4c // 2 + 1
        window = Window(window_length, window="nuttall", norm="none").window
        windows = []
        for i in range(1, n_aperiodicity + 1):
            center = freqency_interval * i * self.fft_length_d4c // sample_rate
            left = center - half_window_length
            right = center + half_window_length + 1
            windows.append(F.pad(window, (left, padded_window_length - right)))
        self.register_buffer("windows", torch.stack(windows))
        self.window_length = window_length

        if fft_length is not None:
            coarse_axis = np.arange(n_aperiodicity + 2) * freqency_interval
            coarse_axis[-1] = sample_rate / 2
            freq_axis = np.arange(fft_length // 2 + 1) * (sample_rate / fft_length)
            idx = np.searchsorted(coarse_axis, freq_axis) - 1
            idx = np.clip(idx, 0, len(coarse_axis) - 2)
            idx = idx.reshape(1, 1, -1)
            self.register_buffer("interp_indices", numpy_to_torch(idx).long())

            x0 = coarse_axis[:-1]
            dx = coarse_axis[1:] - x0
            weights = (freq_axis - np.take(x0, idx)) / np.take(dx, idx)
            self.register_buffer("interp_weights", numpy_to_torch(weights))

        self.spec_love = Spectrum(self.fft_length_love)
        self.spec_d4c = Spectrum(self.fft_length_d4c)

        self.register_buffer("ramp", torch.arange(self.fft_length_d4c))

    def forward(self, x, f0):
        f0 = (
            torch.where(f0 < self.lowest_f0, self.default_f0, f0).unsqueeze(-1).detach()
        )

        # D4CLoveTrain()
        if 0 < self.threshold:
            waveform = get_windowed_waveform(
                x,
                f0,
                3,
                0,
                self.frame_period,
                self.sample_rate,
                self.fft_length_love,
                "blackman",
                False,
                1e-6,
                self.ramp,
            )
            power_spectrum = self.spec_love(waveform)
            rate = self.sample_rate / self.fft_length_love
            boundary0 = math.ceil(100 / rate) + 1
            boundary1 = math.ceil(4000 / rate)
            boundary2 = math.ceil(7900 / rate)
            power_spectrum = torch.cumsum(power_spectrum[..., boundary0:], dim=-1)
            aperiodicity0 = (
                power_spectrum[..., boundary1 - boundary0]
                / power_spectrum[..., boundary2 - boundary0]
            ).unsqueeze(-1)

        # GetCentroid()
        def get_centroid(x, f0, bias_ratio):
            waveform = get_windowed_waveform(
                x,
                f0,
                4,
                bias_ratio,
                self.frame_period,
                self.sample_rate,
                self.fft_length_d4c,
                "blackman",
                False,
                1e-6,
                self.ramp,
            )
            power = waveform.square().sum(dim=-1, keepdim=True)
            waveform = waveform / torch.sqrt(power)
            spectrum1 = torch.fft.rfft(waveform)
            spectrum2 = torch.fft.rfft(waveform * torch.cumsum(waveform != 0, dim=-1))
            centroid = spectrum1.real * spectrum2.real + spectrum1.imag * spectrum2.imag
            return centroid

        # GetStaticCentroid()
        centroid1 = get_centroid(x, f0, -0.25)
        centroid2 = get_centroid(x, f0, 0.25)
        static_centroid = centroid1 + centroid2
        static_centroid = dc_correction(
            static_centroid, f0, self.sample_rate, self.fft_length_d4c, self.ramp
        )

        # GetSmoothedPowerSpectrum()
        waveform = get_windowed_waveform(
            x,
            f0,
            4,
            0,
            self.frame_period,
            self.sample_rate,
            self.fft_length_love,
            "hanning",
            False,
            1e-6,
            self.ramp,
        )
        power_spectrum = self.spec_d4c(waveform)
        power_spectrum = dc_correction(
            power_spectrum, f0, self.sample_rate, self.fft_length_d4c, self.ramp
        )
        smoothed_power_spectrum = linear_smoothing(
            power_spectrum, f0, self.sample_rate, self.fft_length_d4c, self.ramp
        )

        # GetStaticGroupDelay()
        eps = 1e-12
        static_group_delay = static_centroid / (smoothed_power_spectrum + eps)
        static_group_delay = linear_smoothing(
            static_group_delay, f0 / 2, self.sample_rate, self.fft_length_d4c, self.ramp
        )
        smoothed_group_delay = linear_smoothing(
            static_group_delay, f0, self.sample_rate, self.fft_length_d4c, self.ramp
        )
        static_group_delay = static_group_delay - smoothed_group_delay

        # GetCoarseAperiodicity()
        boundary = round(self.fft_length_d4c * 8 / self.window_length)
        power_spectrum = self.spec_d4c(static_group_delay.unsqueeze(-2) * self.windows)
        power_spectrum, _ = torch.sort(power_spectrum)
        power_spectrum = torch.cumsum(power_spectrum, dim=-1)
        coarse_aperiodicity = 10 * torch.log10(
            power_spectrum[..., -(boundary + 2)] / power_spectrum[..., -1]
        )
        coarse_aperiodicity = torch.clip(
            coarse_aperiodicity + (f0 - 100) / 50, max=-eps
        )

        # GetAperiodicity()
        y = coarse_aperiodicity
        if hasattr(self, "interp_indices"):
            y = F.pad(y, (1, 0), value=-60)
            y = F.pad(y, (0, 1), value=-eps)
            y0 = y[..., :-1]
            dy = y[..., 1:] - y0
            index = self.interp_indices.expand(f0.size(0), f0.size(1), -1)
            y = torch.gather(dy, -1, index) * self.interp_weights
            y += torch.gather(y0, -1, index)
        aperiodicity = 10 ** (y / 20)

        if 0 < self.threshold:
            aperiodicity = torch.where(
                aperiodicity0 <= self.threshold, 1 - eps, aperiodicity
            )
        return aperiodicity
