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

# ------------------------------------------------------------------------ #
# Copyright (c) 2013--2022, librosa development team.                      #
#                                                                          #
# Permission to use, copy, modify, and/or distribute this software for any #
# purpose with or without fee is hereby granted, provided that the above   #
# copyright notice and this permission notice appear in all copies.        #
#                                                                          #
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES #
# WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF         #
# MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR  #
# ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES   #
# WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN    #
# ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF  #
# OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.           #
# ------------------------------------------------------------------------ #

import numpy as np
import torch
import torchaudio
from torch import nn

from ..misc.utils import delayed_import
from ..misc.utils import get_resample_params
from ..misc.utils import numpy_to_torch
from .istft import InverseShortTimeFourierTransform as ISTFT


class InverseConstantQTransform(nn.Module):
    """Perform inverse constant-Q transform based on the librosa implementation.

    Parameters
    ----------
    frame_peirod : int >= 1
        Frame period in samples, :math:`P`.

    sample_rate : int >= 1
        Sample rate in Hz.

    f_min : float > 0
        Minimum center frequency in Hz.

    n_bin : int >= 1
        Number of CQ-bins, :math:`K`.

    n_bin_per_octave : int >= 1
        number of bins per octave, :math:`B`.

    tuning : float
        Tuning offset in fractions of a bin.

    filter_scale : float > 0
        Filter scale factor.

    norm : float
        Type of norm used in basis function normalization.

    sparsity : float in [0, 1)
        Sparsification factor.

    window : str
        Window function for the basis.

    scale : bool
        If True, scale the CQT response by the length of filter.

    res_type : ['kaiser_best', 'kaiser_fast'] or None
        Resampling type.

    **kwargs : additional keyword arguments
        See `torchaudio.transforms.Resample
        <https://pytorch.org/audio/main/generated/torchaudio.transforms.Resample.html>`_.

    """

    def __init__(
        self,
        frame_period,
        sample_rate,
        *,
        f_min=32.7,
        n_bin=84,
        n_bin_per_octave=12,
        tuning=0,
        filter_scale=1,
        norm=1,
        sparsity=1e-2,
        window="hann",
        scale=True,
        res_type="kaiser_best",
        **kwargs,
    ):
        super().__init__()

        import librosa

        et_relative_bw = delayed_import("librosa.core.constantq", "__et_relative_bw")
        vqt_filter_fft = delayed_import("librosa.core.constantq", "__vqt_filter_fft")

        assert 1 <= frame_period
        assert 1 <= sample_rate

        K = n_bin
        B = n_bin_per_octave

        n_octave = int(np.ceil(K / B))

        freqs = librosa.cqt_frequencies(
            n_bins=K,
            fmin=f_min,
            bins_per_octave=B,
            tuning=tuning,
        )

        if K == 1:
            alpha = et_relative_bw(B)
        else:
            alpha = librosa.filters._relative_bandwidth(freqs=freqs)

        lengths, _ = librosa.filters.wavelet_lengths(
            freqs=freqs,
            sr=sample_rate,
            window=window,
            filter_scale=filter_scale,
            alpha=alpha,
        )
        if scale:
            cqt_scale = np.sqrt(lengths)
        else:
            cqt_scale = np.ones(K)
        self.register_buffer("cqt_scale", numpy_to_torch(cqt_scale))

        fp = [frame_period]
        sr = [sample_rate]
        for i in range(n_octave - 1):
            if fp[i] % 2 == 0:
                fp.append(fp[i] // 2)
                sr.append(sr[i] * 0.5)
            else:
                fp.append(fp[i])
                sr.append(sr[i])
        fp.reverse()
        sr.reverse()

        slices = []
        transforms = []
        resamplers = []

        if res_type is not None:
            kwargs.update(get_resample_params(res_type))

        for i in range(n_octave):
            n_filter = min(B, K - B * i)
            sl = slice(B * i, B * i + n_filter)
            slices.append(sl)

            fft_basis, fft_length, _ = vqt_filter_fft(
                sr[i],
                freqs[sl],
                filter_scale,
                norm,
                sparsity,
                window=window,
                alpha=alpha[sl],
            )

            fft_basis = np.asarray(fft_basis.conj().todense())
            freq_power = np.reciprocal(np.sum(np.abs(fft_basis) ** 2, axis=1))
            freq_power *= fft_length / lengths[sl]
            fft_basis *= freq_power[:, None]
            self.register_buffer(f"fft_basis_{i}", numpy_to_torch(fft_basis))

            transforms.append(
                ISTFT(
                    frame_length=fft_length,
                    frame_period=fp[i],
                    fft_length=fft_length,
                    center=True,
                    window="rectangular",
                    norm="none",
                )
            )

            resamplers.append(
                torchaudio.transforms.Resample(
                    orig_freq=1,
                    new_freq=sample_rate // sr[i],  # must be integer
                    dtype=torch.get_default_dtype(),
                    **kwargs,
                )
            )

        self.slices = slices
        self.transforms = nn.ModuleList(transforms)
        self.resamplers = nn.ModuleList(resamplers)

    def forward(self, c, out_length=None):
        """Compute inverse constant-Q transform.

        Parameters
        ----------
        c : Tensor [shape=(..., T/P, K)]
            CQT complex output.

        out_length : int or None
            Length of output waveform.

        Returns
        -------
        out : Tensor [shape=(..., T)]
            Reconstructed waveform.

        Examples
        --------
        >>> x = diffsptk.sin(99)
        >>> cqt = diffsptk.CQT(100, 8000, n_bin=4)
        >>> icqt = diffsptk.ICQT(100, 8000, n_bin=4)
        >>> y = icqt(cqt(x), out_length=x.size(0))
        >>> y.shape
        torch.Size([100])

        """
        for i in range(len(self.transforms)):
            C = c[..., self.slices[i]] * self.cqt_scale[self.slices[i]]
            W = getattr(self, f"fft_basis_{i}")
            X = torch.matmul(C, W)
            x = self.transforms[i](X)
            x = self.resamplers[i](x)
            if i == 0:
                y = x[..., :out_length]
            else:
                if out_length is None:
                    end = x.size(-1)
                else:
                    end = min(x.size(-1), out_length)
                y[..., :end] += x[..., :end]
        return y
