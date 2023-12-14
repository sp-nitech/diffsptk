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
import torch.nn.functional as F
import torchaudio

UNVOICED_SYMBOL = 0
TWO_PI = 2 * torch.pi


class Lambda(torch.nn.Module):
    def __init__(self, func, **opt):
        super(Lambda, self).__init__()
        self.func = func
        self.opt = opt

    def forward(self, x):
        return self.func(x, **self.opt)


def is_power_of_two(n):
    return (n != 0) and (n & (n - 1) == 0)


def next_power_of_two(n):
    return 1 << (n - 1).bit_length()


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


def numpy_to_torch(x):
    if isinstance(x, (list, tuple)):
        x = np.array(x)
    if np.iscomplexobj(x):
        return torch.from_numpy(x.astype(default_complex_dtype()))
    else:
        return torch.from_numpy(x.astype(default_dtype()))


def to_3d(x):
    d = x.dim()
    if d == 1:
        y = x.view(1, 1, -1)
    elif d == 2:
        y = x.unsqueeze(1)
    else:
        y = x.view(-1, 1, x.size(-1))
    return y


def get_alpha(sr, mode="hts", n_freq=10, n_alpha=100):
    """Compute frequency warping factor under given sample rate.

    Parameters
    ----------
    sr : int >= 1 [scalar]
        Sample rate in Hz.

    mode : ['hts', 'auto']
        'hts' returns traditional alpha used in HTS. 'auto' computes appropriate
        alpha in L2 sense.

    n_freq : int >= 2 [scalar]
        Number of sample points in the frequency domain.

    n_alpha : int >= 1 [scalar]
        Number of sample points to search alpha.

    Returns
    -------
    alpha : float [0 <= alpha < 1]
        Frequency warping factor, :math:`\\alpha`.

    Examples
    --------
    >>> _, sr = diffsptk.read("assets/data.wav")
    >>> alpha = diffsptk.get_alpha(sr)
    >>> alpha
    0.42

    """

    def get_hts_alpha(sr):
        sr_to_alpha = {
            "8000": 0.31,
            "10000": 0.35,
            "12000": 0.37,
            "16000": 0.42,
            "22050": 0.45,
            "32000": 0.50,
            "44100": 0.53,
            "48000": 0.55,
        }
        key = str(int(sr))
        if key not in sr_to_alpha:
            raise ValueError(f"unsupported sample rate: {sr}")
        selected_alpha = sr_to_alpha[key]
        return selected_alpha

    def get_auto_alpha(sr, n_freq, n_alpha):
        # Compute target mel-frequencies.
        freq = np.arange(n_freq) * (0.5 * sr / (n_freq - 1))
        mel_freq = np.log(1 + freq / 1000)
        mel_freq = mel_freq * (np.pi / mel_freq[-1])
        mel_freq = np.expand_dims(mel_freq, 0)

        # Compute phase characteristic of the 1st order all-pass filter.
        alpha = np.linspace(0, 1, n_alpha, endpoint=False)
        alpha = np.expand_dims(alpha, 1)
        alpha2 = alpha * alpha
        omega = np.arange(n_freq) * (np.pi / (n_freq - 1))
        omega = np.expand_dims(omega, 0)
        numer = (1 - alpha2) * np.sin(omega)
        denom = (1 + alpha2) * np.cos(omega) - 2 * alpha
        warped_omega = np.arctan(numer / denom)
        warped_omega[warped_omega < 0] += np.pi

        # Select an appropriate alpha in terms of L2 distance.
        distance = np.square(mel_freq - warped_omega).sum(axis=1)
        selected_alpha = np.squeeze(alpha[np.argmin(distance)])
        return selected_alpha

    if mode == "hts":
        alpha = get_hts_alpha(sr)
    elif mode == "auto":
        alpha = get_auto_alpha(sr, n_freq, n_alpha)
    else:
        raise ValueError("only hts and auto are supported")

    return alpha


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


def iir(x, b, a):
    """Apply IIR filter.

    Parameters
    ----------
    x : Tensor [shape=(..., B, T) or (..., T)]
        Input signal.

    b : Tensor [shape=(B, M+1) or (M+1,)]
        Numerator coefficients.

    a : Tensor [shape=(B, N+1) or (N+1,)]
        Denominator coefficients.

    Returns
    -------
    y : Tensor [shape=(..., B, T) or (..., T)]
        Output signal.

    """
    diff = b.size(-1) - a.size(-1)
    if 0 < diff:
        a = F.pad(a, (0, diff))
    elif diff < 0:
        b = F.pad(b, (0, -diff))
    y = torchaudio.functional.lfilter(x, a, b, clamp=False, batching=True)
    return y


def deconv1d(x, weight):
    """Deconvolve input. This is not transposed convolution.

    Parameters
    ----------
    x : Tensor [shape=(..., T)]
        Input signal.

    weight : Tensor [shape=(M+1,)]
        Filter coefficients.

    Returns
    -------
    y : Tensor [shape=(..., T-M)]
        Output signal.

    """
    assert weight.dim() == 1
    b = x.view(-1, x.size(-1))
    a = weight.view(1, -1).expand(b.size(0), -1)
    impulse = F.pad(torch.ones_like(b[..., :1]), (0, b.size(-1) - a.size(-1)))
    y = iir(impulse, b, a)
    y = y.view(x.size()[:-1] + y.size()[-1:])
    return y


def check_size(x, y, cause):
    assert x == y, f"Unexpected {cause} (input {x} vs target {y})"


def read(filename, double=False, **kwargs):
    """Read waveform from file.

    Parameters
    ----------
    filename : str [scalar]
        Path of wav file.

    double : bool [scalar]
        If True, return double-type tensor.

    **kwargs : additional keyword arguments
        Additional arguments passed to `soundfile.read`.

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
    x, sr = sf.read(filename, **kwargs)
    if double:
        x = torch.DoubleTensor(x)
    else:
        x = torch.FloatTensor(x)
    return x, sr


def write(filename, x, sr, **kwargs):
    """Write waveform to file.

    Parameters
    ----------
    filename : str [scalar]
        Path of wav file.

    x : Tensor
        Waveform.

    sr : int [scalar]
        Sample rate in Hz.

    **kwargs : additional keyword arguments
        Additional arguments passed to `soundfile.write`.

    Examples
    --------
    >>> x, sr = diffsptk.read("assets/data.wav")
    >>> diffsptk.write("out.wav", x, sr)

    """
    sf.write(filename, x.cpu().numpy(), sr, **kwargs)
