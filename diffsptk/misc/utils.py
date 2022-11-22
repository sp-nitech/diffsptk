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

import warnings

import numpy as np
import soundfile as sf
import torch


def is_power_of_two(n):
    return (n != 0) and (n & (n - 1) == 0)


def next_power_of_two(n):
    return 1 << (n - 1).bit_length()


def is_in(x, ary):
    return any([x == a for a in ary])


def default_dtype():
    t = torch.get_default_dtype()
    if t == torch.float32:  # pragma: no cover
        return np.float32
    elif t == torch.float64:  # pragma: no cover
        return np.float64
    else:
        raise RuntimeError("Unknown default dtype: {t}")


def default_complex_dtype():
    t = torch.get_default_dtype()
    if t == torch.float32:  # pragma: no cover
        return np.complex64
    elif t == torch.float64:  # pragma: no cover
        return np.complex128
    else:
        raise RuntimeError("Unknown default dtype: {t}")


def get_gamma(gamma, c):
    if c is None or c == 0:
        return gamma
    if gamma != 0:
        warnings.warn("gamma is given, but will be ignored")
    assert 1 <= c
    return -1 / c


def symmetric_toeplitz(x):
    d = x.size(-1)
    xx = torch.cat((x[..., 1:].flip(-1), x), dim=-1)
    X = xx.unfold(-1, d, 1).flip(-2)
    return X


def hankel(x):
    d = x.size(-1)
    assert d % 2 == 1
    X = x.unfold(-1, (d + 1) // 2, 1)
    return X


def cexp(x):
    return torch.polar(torch.exp(x.real), x.imag)


def clog(x):
    return torch.log(x.abs())


def check_size(x, y, cause):
    assert x == y, f"Unexpected {cause} (input {x} vs target {y})"


def read(filename, double=False):
    """Read waveform from file.

    Parameters
    ----------
    filename : str [scalar]
        Path of wav file.

    double : bool [scalar]
        If True, return double-type tensor.

    Returns
    -------
    x : Tensor
        Waveform.

    Examples
    --------
    >>> x, sr = diffsptk.read("assets/data.wav")
    >>> x
    tensor([ 0.0002,  0.0004,  0.0006,  ...,  0.0006, -0.0006, -0.0007])
    >>> sr
    16000

    """
    x, sr = sf.read(filename)
    if double:
        x = torch.DoubleTensor(x)
    else:
        x = torch.FloatTensor(x)
    return x, sr


def write(filename, x, sr):
    """Write waveform to file.

    Parameters
    ----------
    filename : str [scalar]
        Path of wav file.

    x : Tensor
        Waveform.

    sr : int [scalar]
        Sample rate in Hz.

    Examples
    --------
    >>> x, sr = diffsptk.read("assets/data.wav")
    >>> diffsptk.write("out.wav", x, sr)

    """
    sf.write(filename, x.cpu().numpy(), sr)
