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

import librosa
import numpy as np
import torch
import torch.nn as nn
import torchaudio

from ..misc.utils import numpy_to_torch
from .stft import ShortTimeFourierTransform as STFT

_vqt_filter_fft = librosa.core.constantq.__vqt_filter_fft
_bpo_to_alpha = librosa.core.constantq.__bpo_to_alpha


class ConstantQTransform(nn.Module):
    """Perform constant-Q transform based on the librosa implementation.

    Parameters
    ----------
    frame_peirod : int >= 1
        Frame period in samples.

    sample_rate : int >= 1
        Sample rate in Hz.

    f_min : float > 0
        Minimum center frequency in Hz.

    n_bin : int >= 1
        Number of CQ-bins.

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
        If True, scale the CQT responce by the length of filter.

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
        sparsity=0.01,
        window="hann",
        scale=True,
        **kwargs,
    ):
        super(ConstantQTransform, self).__init__()

        assert 1 <= frame_period
        assert 1 <= sample_rate

        self.frame_period = frame_period
        self.sample_rate = sample_rate

        n_octave = int(np.ceil(n_bin / n_bin_per_octave))
        n_filter = min(n_bin_per_octave, n_bin)

        f_min = f_min * 2 ** (tuning / n_bin_per_octave)

        freqs = librosa.cqt_frequencies(
            n_bins=n_bin,
            fmin=f_min,
            bins_per_octave=n_bin_per_octave,
        )

        alpha = _bpo_to_alpha(n_bin_per_octave)

        if scale:
            lengths, _ = librosa.filters.wavelet_lengths(
                freqs=freqs,
                sr=sample_rate,
                window=window,
                filter_scale=filter_scale,
                alpha=alpha,
            )
            cqt_scale = 1 / np.sqrt(lengths)
        else:
            cqt_scale = np.ones(n_bin)
        self.register_buffer("cqt_scale", numpy_to_torch(cqt_scale))

        transforms = []
        resamplers = []

        fp = frame_period
        sr = sample_rate
        for i in range(n_octave):
            if i == 0:
                sl = slice(-n_filter, None)
            else:
                sl = slice(-n_filter * (i + 1), -n_filter * i)

            fft_basis, n_fft, _ = _vqt_filter_fft(
                sr,
                freqs[sl],
                filter_scale,
                norm,
                sparsity,
                window=window,
                alpha=alpha,
            )

            fft_basis[:] *= np.sqrt(sample_rate / sr)
            self.register_buffer(
                f"fft_basis_{i}", numpy_to_torch(fft_basis.todense()).T
            )

            transforms.append(
                STFT(
                    frame_length=n_fft,
                    frame_period=fp,
                    fft_length=n_fft,
                    center=True,
                    window="rectangular",
                    norm="none",
                    eps=0,
                    out_format="complex",
                )
            )

            if fp % 2 == 0:
                fp //= 2
                sr /= 2
                resamplers.append(
                    torchaudio.transforms.Resample(
                        orig_freq=2,
                        new_freq=1,
                        dtype=torch.get_default_dtype(),
                        **kwargs,
                    )
                )

        self.transforms = nn.ModuleList(transforms)
        self.resamplers = nn.ModuleList(resamplers)
        self.resample_scale = 1 / np.sqrt(0.5)

    def forward(self, x):
        """Apply CQT to signal.

        Parameters
        ----------
        x : Tensor [shape=(..., T)]
            Signal.

        Returns
        -------
        Tensor [shape=(..., N, K)]
            CQT complex output, where N is the number of frames and K is CQ-bin.

        Examples
        --------
        >>> x = diffsptk.sin(99)
        >>> cqt = diffsptk.CQT(100, 8000, n_bin=4)
        >>> c = cqt(x).abs()
        >>> c
        tensor([[1.1259, 1.2069, 1.3008, 1.3885]])

        """
        c = []
        fp = self.frame_period
        for i in range(len(self.transforms)):
            X = self.transforms[i](x)
            W = getattr(self, f"fft_basis_{i}")
            c.append(torch.matmul(X, W))
            if fp % 2 == 0:
                fp //= 2
                x = self.resamplers[i](x) * self.resample_scale
        c = self._trim_stack(c) * self.cqt_scale
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
