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
import scipy


def normalize(
    S,
    *,
    norm=np.inf,
    axis=0,
    threshold=None,
    fill=None,
):
    # Avoid div-by-zero
    if threshold is None:
        threshold = tiny(S)

    elif threshold <= 0:
        raise ValueError(f"threshold={threshold} must be strictly positive")

    if fill not in [None, False, True]:
        raise ValueError(f"fill={fill} must be None or boolean")

    if not np.all(np.isfinite(S)):
        raise ValueError("Input must be finite")

    # All norms only depend on magnitude, let's do that first
    mag = np.abs(S).astype(float)

    # For max/min norms, filling with 1 works
    fill_norm = 1

    if norm is None:
        return S

    elif norm == np.inf:
        length = np.max(mag, axis=axis, keepdims=True)

    elif norm == -np.inf:
        length = np.min(mag, axis=axis, keepdims=True)

    elif norm == 0:
        if fill is True:
            raise ValueError("Cannot normalize with norm=0 and fill=True")

        length = np.sum(mag > 0, axis=axis, keepdims=True, dtype=mag.dtype)

    elif np.issubdtype(type(norm), np.number) and norm > 0:
        length = np.sum(mag**norm, axis=axis, keepdims=True) ** (1.0 / norm)

        if axis is None:
            fill_norm = mag.size ** (-1.0 / norm)
        else:
            fill_norm = mag.shape[axis] ** (-1.0 / norm)

    else:
        raise ValueError(f"Unsupported norm: {repr(norm)}")

    # indices where norm is below the threshold
    small_idx = length < threshold

    Snorm = np.empty_like(S)
    if fill is None:
        # Leave small indices un-normalized
        length[small_idx] = 1.0
        Snorm[:] = S / length

    elif fill:
        # If we have a non-zero fill value, we locate those entries by
        # doing a nan-divide.
        # If S was finite, then length is finite (except for small positions)
        length[small_idx] = np.nan
        Snorm[:] = S / length
        Snorm[np.isnan(Snorm)] = fill_norm
    else:
        # Set small values to zero by doing an inf-divide.
        # This is safe (by IEEE-754) as long as S is finite.
        length[small_idx] = np.inf
        Snorm[:] = S / length

    return Snorm


def pad_center(data, *, size, axis=-1, **kwargs):
    kwargs.setdefault("mode", "constant")

    n = data.shape[axis]

    lpad = int((size - n) // 2)

    lengths = [(0, 0)] * data.ndim
    lengths[axis] = (lpad, int(size - n - lpad))

    if lpad < 0:
        raise ValueError(f"Target size ({size:d}) must be at least input size ({n:d})")

    return np.pad(data, lengths, **kwargs)


def phasor(angles, *, mag=None):
    z = phasor_angles(angles)

    if mag is not None:
        z *= mag

    return z


def phasor_angles(x):
    return np.cos(x) + 1j * np.sin(x)


def sparsify_rows(x, *, quantile=0.01, dtype=None):
    if x.ndim == 1:
        x = x.reshape((1, -1))

    elif x.ndim > 2:
        raise ValueError(
            f"Input must have 2 or fewer dimensions. Provided x.shape={x.shape}."
        )

    if not 0.0 <= quantile < 1:
        raise ValueError(f"Invalid quantile {quantile:.2f}")

    if dtype is None:
        dtype = x.dtype

    x_sparse = scipy.sparse.lil_matrix(x.shape, dtype=dtype)

    mags = np.abs(x)
    norms = np.sum(mags, axis=1, keepdims=True)

    mag_sort = np.sort(mags, axis=1)
    cumulative_mag = np.cumsum(mag_sort / norms, axis=1)

    threshold_idx = np.argmin(cumulative_mag < quantile, axis=1)

    for i, j in enumerate(threshold_idx):
        idx = np.where(mags[i] >= mag_sort[i, j])
        x_sparse[i, idx] = x[i, idx]

    return x_sparse.tocsr()


def tiny(x):
    # Make sure we have an array view
    x = np.asarray(x)

    # Only floating types generate a tiny
    if np.issubdtype(x.dtype, np.floating) or np.issubdtype(
        x.dtype, np.complexfloating
    ):
        dtype = x.dtype
    else:
        dtype = np.dtype(np.float32)

    return np.finfo(dtype).tiny
