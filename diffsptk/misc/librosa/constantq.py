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

from .filters import wavelet
from .util import sparsify_rows


def cqt_frequencies(n_bins, *, fmin, bins_per_octave=12, tuning=0.0):
    correction: float = 2.0 ** (float(tuning) / bins_per_octave)
    frequencies: np.ndarray = 2.0 ** (
        np.arange(0, n_bins, dtype=float) / bins_per_octave
    )

    return correction * fmin * frequencies


def early_downsample_count(nyquist, filter_cutoff, hop_length, n_octaves):
    downsample_count1 = max(0, int(np.ceil(np.log2(nyquist / filter_cutoff)) - 1) - 1)

    num_twos = num_two_factors(hop_length)
    downsample_count2 = max(0, num_twos - n_octaves + 1)

    return min(downsample_count1, downsample_count2)


def et_relative_bw(bins_per_octave):
    r = 2 ** (1 / bins_per_octave)
    return np.atleast_1d((r**2 - 1) / (r**2 + 1))


def num_two_factors(x):
    if x <= 0:
        return 0
    num_twos = 0
    while x % 2 == 0:
        num_twos += 1
        x //= 2

    return num_twos


def vqt_filter_fft(
    sr,
    freqs,
    filter_scale,
    norm,
    sparsity,
    hop_length=None,
    window="hann",
    gamma=0.0,
    dtype=np.complex64,
    alpha=None,
):
    basis, lengths = wavelet(
        freqs=freqs,
        sr=sr,
        filter_scale=filter_scale,
        norm=norm,
        pad_fft=True,
        window=window,
        gamma=gamma,
        alpha=alpha,
    )

    # Filters are padded up to the nearest integral power of 2
    n_fft = basis.shape[1]

    if hop_length is not None and n_fft < 2.0 ** (1 + np.ceil(np.log2(hop_length))):
        n_fft = int(2.0 ** (1 + np.ceil(np.log2(hop_length))))

    # re-normalize bases with respect to the FFT window length
    basis *= lengths[:, np.newaxis] / float(n_fft)

    # FFT and retain only the non-negative frequencies
    fft_basis = np.fft.fft(basis, n=n_fft, axis=1)[:, : (n_fft // 2) + 1]

    # sparsify the basis
    fft_basis = sparsify_rows(fft_basis, quantile=sparsity, dtype=dtype)

    return fft_basis, n_fft, lengths
