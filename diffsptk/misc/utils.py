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

from importlib import import_module
import logging
import math

import numpy as np
import soundfile as sf
import torch
from torch import nn
import torch.nn.functional as F
import torchaudio

UNVOICED_SYMBOL = 0
TWO_PI = math.tau


class Lambda(nn.Module):
    def __init__(self, func, **opt):
        super().__init__()
        self.func = func
        self.opt = opt

    def forward(self, x):
        return self.func(x, **self.opt)


def delayed_import(module_path, item_name):
    module = import_module(module_path)
    return getattr(module, item_name)


def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s"
    )
    logger.handlers.clear()
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def get_generator(seed=None):
    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)
    return generator


def next_power_of_two(n):
    return 1 << (n - 1).bit_length()


def default_dtype():
    t = torch.get_default_dtype()
    if t == torch.float:
        return np.float32
    elif t == torch.double:
        return np.float64
    raise RuntimeError("Unknown default dtype: {t}.")


def default_complex_dtype():
    t = torch.get_default_dtype()
    if t == torch.float:
        return np.complex64
    elif t == torch.double:
        return np.complex128
    raise RuntimeError("Unknown default dtype: {t}.")


def torch_default_complex_dtype():
    t = torch.get_default_dtype()
    if t == torch.float:
        return torch.complex64
    elif t == torch.double:
        return torch.complex128
    raise RuntimeError("Unknown default dtype: {t}.")


def numpy_to_torch(x):
    if np.iscomplexobj(x):
        return torch.from_numpy(x.astype(default_complex_dtype()))
    else:
        return torch.from_numpy(x.astype(default_dtype()))


def to(x, dtype=None, device=None):
    if dtype is None:
        if torch.is_complex(x):
            dtype = torch_default_complex_dtype()
        else:
            dtype = torch.get_default_dtype()
    return x.to(dtype=dtype, device=device)


def to_2d(x):
    y = x.view(-1, x.size(-1))
    return y


def to_3d(x):
    y = x.view(-1, 1, x.size(-1))
    return y


def to_dataloader(x, batch_size=None):
    if torch.is_tensor(x):
        dataset = torch.utils.data.TensorDataset(x)
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=len(x) if batch_size is None else batch_size,
            shuffle=False,
            drop_last=False,
        )
        return data_loader
    elif isinstance(x, torch.utils.data.DataLoader):
        return x
    else:
        raise ValueError(f"Unsupported input type: {type(x)}.")


def reflect(x):
    d = x.size(-1)
    y = x.view(-1, d)
    y = F.pad(y, (d - 1, 0), mode="reflect")
    y = y.view(*x.size()[:-1], -1)
    return y


def replicate1(x, left=True, right=True):
    d = x.size(-1)
    y = x.view(-1, d)
    y = F.pad(y, (1 if left else 0, 1 if right else 0), mode="replicate")
    y = y.view(*x.size()[:-1], -1)
    return y


def remove_gain(a, value=1, return_gain=False):
    K, a1 = torch.split(a, [1, a.size(-1) - 1], dim=-1)
    a = F.pad(a1, (1, 0), value=value)
    if return_gain:
        ret = (K, a)
    else:
        ret = a
    return ret


def get_resample_params(mode="kaiser_best"):
    # From https://pytorch.org/audio/stable/tutorials/audio_resampling_tutorial.html
    if mode == "kaiser_best":
        params = {
            "lowpass_filter_width": 64,
            "rolloff": 0.9475937167399596,
            "resampling_method": "sinc_interp_kaiser",
            "beta": 14.769656459379492,
        }
    elif mode == "kaiser_fast":
        params = {
            "lowpass_filter_width": 16,
            "rolloff": 0.85,
            "resampling_method": "sinc_interp_kaiser",
            "beta": 8.555504641634386,
        }
    else:
        raise ValueError("Only kaiser_best and kaiser_fast are supported.")
    return params


def get_alpha(sr, mode="hts", n_freq=10, n_alpha=100):
    """Compute an appropriate frequency warping factor under given sample rate.

    Parameters
    ----------
    sr : int >= 1
        Sample rate in Hz.

    mode : ['hts', 'auto']
        'hts' returns traditional alpha used in HTS. 'auto' computes appropriate
        alpha in L2 sense.

    n_freq : int >= 2
        Number of sample points in the frequency domain.

    n_alpha : int >= 1
        Number of sample points to search alpha.

    Returns
    -------
    out : float in [0, 1)
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
            raise ValueError(f"Unsupported sample rate: {sr}.")
        selected_alpha = sr_to_alpha[key]
        return selected_alpha

    def get_auto_alpha(sr, n_freq, n_alpha):
        # Compute target mel-frequencies.
        freq = np.arange(n_freq) * (0.5 * sr / (n_freq - 1))
        mel_freq = np.log1p(freq / 1000)
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
        selected_alpha = float(np.squeeze(alpha[np.argmin(distance)]))
        return selected_alpha

    if mode == "hts":
        alpha = get_hts_alpha(sr)
    elif mode == "auto":
        alpha = get_auto_alpha(sr, n_freq, n_alpha)
    else:
        raise ValueError("Only hts and auto are supported.")

    return alpha


def get_gamma(gamma, c):
    if c is None or c == 0:
        return gamma
    assert 1 <= c
    return -1 / c


def symmetric_toeplitz(x):
    d = x.size(-1)
    xx = reflect(x)
    X = xx.unfold(-1, d, 1).flip(-2)
    return X


def hankel(x):
    d = x.size(-1)
    n = (d + 1) // 2
    X = x.unfold(-1, n, 1)[..., :n, :]
    return X


def vander(x):
    X = torch.linalg.vander(x).transpose(-2, -1)
    return X


def cas(x):
    return (2**0.5) * torch.cos(x - 0.25 * torch.pi)  # cos(x) + sin(x)


def cexp(x):
    return torch.polar(torch.exp(x.real), x.imag)


def clog(x):
    return torch.log(x.abs())


def outer(x, y=None):
    return torch.matmul(
        x.unsqueeze(-1), x.unsqueeze(-2) if y is None else y.unsqueeze(-2)
    )


def iir(x, b=None, a=None):
    if b is None:
        b = torch.ones(1, dtype=x.dtype, device=x.device)
    if a is None:
        a = torch.ones(1, dtype=x.dtype, device=x.device)

    diff = b.size(-1) - a.size(-1)
    if 0 < diff:
        a = F.pad(a, (0, diff))
    elif diff < 0:
        b = F.pad(b, (0, -diff))
    y = torchaudio.functional.lfilter(x, a, b, clamp=False, batching=True)
    return y


def plateau(length, first, middle, last=None, dtype=None, device=None):
    x = torch.full((length,), middle, dtype=dtype, device=device)
    x[0] = first
    if last is not None:
        x[-1] = last
    return x


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
    out : Tensor [shape=(..., T-M)]
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
    assert x == y, f"Unexpected {cause} (input {x} vs target {y})."


def read(filename, double=False, device=None, **kwargs):
    """Read waveform from file.

    Parameters
    ----------
    filename : str
        Path of wav file.

    double : bool
        If True, return double-type tensor.

    device : torch.device or None
        Device of returned tensor.

    **kwargs : additional keyword arguments
        Additional arguments passed to `soundfile.read
        <https://python-soundfile.readthedocs.io/en/latest/#soundfile.read>`_.

    Returns
    -------
    x : Tensor
        Waveform.

    sr : int
        Sample rate in Hz.

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
    if device is not None:
        x = x.to(device)
    return x, sr


def write(filename, x, sr, **kwargs):
    """Write waveform to file.

    Parameters
    ----------
    filename : str
        Path of wav file.

    x : Tensor
        Waveform.

    sr : int
        Sample rate in Hz.

    **kwargs : additional keyword arguments
        Additional arguments passed to `soundfile.write
        <https://python-soundfile.readthedocs.io/en/latest/#soundfile.write>`_.

    Examples
    --------
    >>> x, sr = diffsptk.read("assets/data.wav")
    >>> diffsptk.write("out.wav", x, sr)

    """
    x = x.cpu().numpy() if torch.is_tensor(x) else x
    sf.write(filename, x, sr, **kwargs)
