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

import logging
import math
from itertools import islice

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from torch import nn

UNVOICED_SYMBOL = 0
TAU = math.tau


class Lambda(nn.Module):
    def __init__(self, func, **opt):
        super().__init__()
        self.func = func
        self.opt = opt

    def forward(self, x):
        return self.func(x, **self.opt)


def get_layer(is_module, module, params):
    if is_module:
        return module(**params)

    if module._takes_input_size():
        params = dict(islice(params.items(), 1, None))

    def layer(*args, **kwargs):
        return module._func(*args, **params, **kwargs)

    return layer


def get_values(dictionary, begin=1, end=-1):
    return list(dictionary.values())[begin:end]


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


def check_size(x, y, cause):
    if x != y:
        raise ValueError(f"Unexpected {cause} (input {x} vs target {y}).")


def is_power_of_two(n):
    return (n != 0) and (n & (n - 1) == 0)


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


def to(x, device=None, dtype=None):
    if dtype is None:
        if torch.is_complex(x):
            dtype = torch_default_complex_dtype()
        else:
            dtype = torch.get_default_dtype()
    return x.to(device=device, dtype=dtype)


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


def get_gamma(gamma, c):
    if c is None or c == 0:
        return gamma
    if not 1 <= c:
        raise ValueError("c must be an integer greater than or equal to 1.")
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


def iir(x, b=None, a=None, batching=True):
    if b is None:
        b = torch.ones(1, device=x.device, dtype=x.dtype)
    if a is None:
        a = torch.ones(1, device=x.device, dtype=x.dtype)

    diff = b.size(-1) - a.size(-1)
    if 0 < diff:
        a = F.pad(a, (0, diff))
    elif diff < 0:
        b = F.pad(b, (0, -diff))
    y = torchaudio.functional.lfilter(x, a, b, clamp=False, batching=batching)
    return y


def plateau(length, first, middle, last=None, device=None, dtype=None):
    x = torch.full((length,), middle, device=device, dtype=dtype)
    x[0] = first
    if last is not None:
        x[-1] = last
    return x


def deconv1d(x, weight):
    """Deconvolve the input signal. Note that this is not transposed convolution.

    Parameters
    ----------
    x : Tensor [shape=(..., T)]
        The input signal.

    weight : Tensor [shape=(M+1,)]
        The filter coefficients.

    Returns
    -------
    out : Tensor [shape=(..., T-M)]
        The output signal.

    """
    if weight.dim() != 1:
        raise ValueError("The weight must be 1D.")
    b = x.view(-1, x.size(-1))
    a = weight.view(1, -1).expand(b.size(0), -1)
    impulse = F.pad(torch.ones_like(b[..., :1]), (0, b.size(-1) - a.size(-1)))
    y = iir(impulse, b, a)
    y = y.view(x.size()[:-1] + y.size()[-1:])
    return y
