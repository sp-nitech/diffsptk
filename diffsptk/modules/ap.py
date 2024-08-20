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
from torch import nn
import torch.nn.functional as F

from ..misc.utils import UNVOICED_SYMBOL
from ..misc.utils import numpy_to_torch


class Aperiodicity(nn.Module):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/ap.html>`_
    for details.

    Parameters
    ----------
    frame_period : int >= 1
        Frame period, :math:`P`.

    sample_rate : int >= 1
        Sample rate in Hz.

    fft_length : int
        Size of double-sided aperiodicity, :math:`L`.

    algorithm : ['tandem']
        Algorithm.

    out_format : ['a', 'p', 'a/p', 'p/a']
        Output format.

    lower_bound : float >= 0
        Lower bound of aperiodicity.

    upper_bound : float <= 1
        Upper bound of aperiodicity.

    window_length_ms : int >= 1
        Window length in msec.

    eps : float > 0
        A number used to stabilize colesky decomposition.

    """

    def __init__(
        self,
        frame_period,
        sample_rate,
        fft_length,
        algorithm="tandem",
        out_format="a",
        **kwargs,
    ):
        super().__init__()

        assert 1 <= frame_period
        assert 1 <= sample_rate

        if algorithm == "tandem":
            self.extractor = AperiodicityExtractionByTandem(
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
            Waveform.

        f0 : Tensor [shape=(B, N) or (N,)]
            F0 in Hz.

        Returns
        -------
        out : Tensor [shape=(B, N, L/2+1) or (N, L/2+1)]
            Aperiodicity.

        Examples
        --------
        >>> x = diffsptk.sin(100, 10)
        >>> pitch = diffsptk.Pitch(80, 16000, out_format="f0")
        >>> f0 = pitch(x)
        >>> f0
        tensor([1597.2064, 1597.2064])
        >>> aperiodicity = diffsptk.Aperiodicity(80, 16000, 8)
        >>> ap = aperiodicity(x, f0)
        >>> ap
        tensor([[0.0010, 0.0010, 0.1729, 0.1647, 0.1569],
                [0.0010, 0.0010, 0.0490, 0.0487, 0.0483]])

        """
        d = x.dim()
        if d == 1:
            x = x.unsqueeze(0)
        assert x.dim() == 2, "Input must be 2D tensor."

        if f0.dim() == 1:
            f0 = f0.unsqueeze(0)
        assert f0.dim() == 2, "Input must be 2D tensor."

        ap = self.convert(self.extractor(x, f0))

        if d == 1:
            ap = ap.squeeze(0)
        return ap


class AperiodicityExtractionByTandem(nn.Module):
    """Aperiodicity extraction by TANDEM-STRAIGHT."""

    def __init__(
        self,
        frame_period,
        sample_rate,
        fft_length,
        lower_bound=0.001,
        upper_bound=0.999,
        window_length_ms=30,
        eps=1e-5,
    ):
        super().__init__()

        assert fft_length % 2 == 0
        assert 0 <= lower_bound < upper_bound <= 1
        assert 1 <= window_length_ms
        assert 0 < eps

        self.frame_period = frame_period
        self.sample_rate = sample_rate
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.n_band = int(np.log2(sample_rate / 600))
        assert self.n_band <= fft_length // 2

        self.default_f0 = 150
        self.n_trial = 10

        self.cutoff_list = [sample_rate / 2**i for i in range(2, self.n_band + 1)]
        self.cutoff_list.append(self.cutoff_list[-1])

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
        f0 = f0.detach().clone()
        f0[f0 == UNVOICED_SYMBOL] = self.default_f0

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
        bap = torch.stack(bap[::-1], dim=-1)  # (B, N, D)
        bap = torch.clip(bap, min=self.lower_bound, max=self.upper_bound)

        # Interpolate band aperiodicity.
        y = torch.log(bap)
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
