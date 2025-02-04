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

from ..misc.utils import Lambda
from ..misc.utils import delayed_import
from ..misc.utils import get_resample_params
from ..misc.utils import numpy_to_torch
from .stft import ShortTimeFourierTransform as STFT


class ConstantQTransform(nn.Module):
    """Perform constant-Q transform based on the librosa implementation.

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
        early_downsample_count = delayed_import(
            "librosa.core.constantq", "__early_downsample_count"
        )

        assert 1 <= frame_period
        assert 1 <= sample_rate

        K = n_bin
        B = n_bin_per_octave

        n_octave = int(np.ceil(K / B))
        n_filter = min(B, K)

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

        lengths, filter_cutoff = librosa.filters.wavelet_lengths(
            freqs=freqs,
            sr=sample_rate,
            window=window,
            filter_scale=filter_scale,
            alpha=alpha,
        )

        early_downsample = []
        downsample_count = early_downsample_count(
            sample_rate * 0.5, filter_cutoff, frame_period, n_octave
        )
        if res_type is not None:
            kwargs.update(get_resample_params(res_type))
        if 0 < downsample_count:
            downsample_factor = 2**downsample_count
            early_downsample.append(
                torchaudio.transforms.Resample(
                    orig_freq=downsample_factor,
                    new_freq=1,
                    dtype=torch.get_default_dtype(),
                    **kwargs,
                )
            )
            if scale:
                downsample_scale = np.sqrt(downsample_factor)
            else:
                downsample_scale = downsample_factor
            early_downsample.append(Lambda(lambda x: x * downsample_scale))

            # Update frame period and sample rate.
            frame_period //= downsample_factor
            sample_rate /= downsample_factor

            # Update lengths for scaling.
            if scale:
                lengths, _ = librosa.filters.wavelet_lengths(
                    freqs=freqs,
                    sr=sample_rate,
                    window=window,
                    filter_scale=filter_scale,
                    alpha=alpha,
                )
        self.early_downsample = nn.Sequential(*early_downsample)

        if scale:
            cqt_scale = np.reciprocal(np.sqrt(lengths))
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

        transforms = []
        resamplers = []

        for i in range(n_octave):
            sl = slice(-n_filter * (i + 1), None if i == 0 else (-n_filter * i))

            fft_basis, fft_length, _ = vqt_filter_fft(
                sr[i],
                freqs[sl],
                filter_scale,
                norm,
                sparsity,
                window=window,
                alpha=alpha[sl],
            )

            fft_basis *= np.sqrt(sample_rate / sr[i])
            self.register_buffer(
                f"fft_basis_{i}", numpy_to_torch(fft_basis.todense()).T
            )

            transforms.append(
                STFT(
                    frame_length=fft_length,
                    frame_period=fp[i],
                    fft_length=fft_length,
                    center=True,
                    window="rectangular",
                    norm="none",
                    eps=0,
                    out_format="complex",
                )
            )

            if fp[i] % 2 == 0:
                resample_scale = np.sqrt(2)
                resamplers.append(
                    nn.Sequential(
                        torchaudio.transforms.Resample(
                            orig_freq=2,
                            new_freq=1,
                            dtype=torch.get_default_dtype(),
                            **kwargs,
                        ),
                        Lambda(lambda x: x * resample_scale),
                    )
                )
            else:
                resamplers.append(Lambda(lambda x: x))

        self.transforms = nn.ModuleList(transforms)
        self.resamplers = nn.ModuleList(resamplers)

    def forward(self, x):
        """Compute constant-Q transform.

        Parameters
        ----------
        x : Tensor [shape=(..., T)]
            Waveform.

        Returns
        -------
        out : Tensor [shape=(..., T/P, K)]
            CQT complex output.

        Examples
        --------
        >>> x = diffsptk.sin(99)
        >>> cqt = diffsptk.CQT(100, 8000, n_bin=4)
        >>> c = cqt(x).abs()
        >>> c
        tensor([[1.1259, 1.2069, 1.3008, 1.3885]])

        """
        x = self.early_downsample(x)

        cs = []
        for i in range(len(self.transforms)):
            X = self.transforms[i](x)
            W = getattr(self, f"fft_basis_{i}")
            cs.append(torch.matmul(X, W))
            if i != len(self.transforms) - 1:
                x = self.resamplers[i](x)
        c = self._trim_stack(cs) * self.cqt_scale
        return c

    def _trim_stack(self, cqt_response):
        max_col = min(c.shape[-2] for c in cqt_response)
        n_bin = len(self.cqt_scale)
        shape = list(cqt_response[0].shape)
        shape[-2] = max_col
        shape[-1] = n_bin
        output = torch.empty(
            shape, dtype=cqt_response[0].dtype, device=cqt_response[0].device
        )

        end = n_bin
        for c in cqt_response:
            n_octave = c.shape[-1]
            if end < n_octave:
                output[..., :end] = c[..., :max_col, -end:]
            else:
                output[..., end - n_octave : end] = c[..., :max_col, :]
            end -= n_octave
        return output
