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
import torchaudio
from torch import nn

from ..misc.librosa import cqt_frequencies
from ..misc.librosa import et_relative_bw
from ..misc.librosa import relative_bandwidth
from ..misc.librosa import vqt_filter_fft
from ..misc.librosa import wavelet_lengths
from ..misc.utils import get_resample_params
from ..misc.utils import numpy_to_torch
from .base import BaseNonFunctionalModule
from .istft import InverseShortTimeFourierTransform as ISTFT


class InverseConstantQTransform(BaseNonFunctionalModule):
    """Perform inverse constant-Q transform based on the librosa implementation.

    Parameters
    ----------
    frame_period : int >= 1
        The frame period in samples, :math:`P`.

    sample_rate : int >= 1
        The sample rate in Hz.

    f_min : float > 0
        The minimum center frequency in Hz.

    n_bin : int >= 1
        The number of CQ-bins, :math:`K`.

    n_bin_per_octave : int >= 1
        The number of bins per octave, :math:`B`.

    tuning : float
        The tuning offset in fractions of a bin.

    filter_scale : float > 0
        The filter scale factor.

    norm : float
        The type of norm used in the basis function normalization.

    sparsity : float in [0, 1)
        The sparsification factor.

    window : str
        The window function for the basis.

    scale : bool
        If True, scale the CQT response by the length of the filter.

    res_type : ['kaiser_best', 'kaiser_fast'] or None
        The resampling type.

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

        if frame_period <= 0:
            raise ValueError("frame_period must be positive.")

        K = n_bin
        B = n_bin_per_octave

        n_octave = int(np.ceil(K / B))

        freqs = cqt_frequencies(
            n_bins=K,
            fmin=f_min,
            bins_per_octave=B,
            tuning=tuning,
        )

        if K == 1:
            alpha = et_relative_bw(B)
        else:
            alpha = relative_bandwidth(freqs=freqs)

        lengths, _ = wavelet_lengths(
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
            The CQT complex input.

        out_length : int or None
            The length of the output waveform.

        Returns
        -------
        out : Tensor [shape=(..., T)]
            The reconstructed waveform.

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
                end = x.size(-1) if out_length is None else min(x.size(-1), out_length)
                y[..., :end] += x[..., :end]
        return y
