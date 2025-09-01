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

import functools
import subprocess
import time
import warnings
from itertools import islice

import numpy as np
import soundfile as sf
import torch


def is_array(x):
    return isinstance(x, (list, tuple))


def compose(*fs):
    def compose2_outer_kwargs(f, g):
        return lambda *args, **kwargs: f(g(*args), **kwargs)

    return functools.reduce(compose2_outer_kwargs, fs)


def choice(is_module, module, func, params={}):
    if is_module:
        return module(**params)

    exclude_keys = ("learnable", "device", "dtype")
    filtered_params = {k: v for k, v in params.items() if k not in exclude_keys}
    if module._takes_input_size():
        filtered_params = dict(islice(filtered_params.items(), 1, None))

    def f(*args, **kwargs):
        return func(*args, **filtered_params, **kwargs)

    return f


def dtype_to_complex_dtype(dtype: torch.dtype | None) -> torch.dtype:
    if dtype is None:
        dtype = torch.get_default_dtype()
    if dtype == torch.double:
        dtype = torch.complex128
    elif dtype == torch.float:
        dtype = torch.complex64
    else:
        raise ValueError(f"Unsupported dtype: {dtype}.")
    return dtype


def allclose(a, b, rtol=None, atol=None, dtype=None, factor=1):
    is_double = dtype == torch.double
    if rtol is None:
        rtol = (1e-5 if is_double else 1e-4) * factor
    if atol is None:
        atol = (1e-8 if is_double else 1e-6) * factor
    return np.allclose(a, b, rtol=rtol, atol=atol)


def call(cmd, get=True):
    if get:
        res = subprocess.run(
            cmd + " | x2x +da -f %.15g",
            shell=True,
            text=True,
            stdout=subprocess.PIPE,
            check=False,
        )
        data = np.fromstring(res.stdout, sep="\n", dtype=np.double)
        if len(data) == 0:
            raise RuntimeError(f"Failed to run command: {cmd}")
        return data
    else:
        res = subprocess.run(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            check=False,
        )
        if res.returncode != 0:
            raise RuntimeError(f"Failed to run command: {cmd}")
        return None


def check_compatibility(
    device,
    dtype,
    modules,
    setup,
    inputs,
    target,
    teardown,
    dx=None,
    dy=None,
    eq=None,
    opt={},
    get=None,
    key=[],
    sr=None,
    verbose=False,
    **kwargs,
):
    if dtype is None:
        dtype = torch.get_default_dtype()

    for cmd in setup:
        call(cmd, get=False)

    if not is_array(modules):
        modules = [modules]
    if not is_array(inputs):
        inputs = [inputs]

    x = []
    for i, cmd in enumerate(inputs):
        x.append(torch.from_numpy(call(cmd)).to(device=device, dtype=dtype))
        if is_array(dx):
            if dx[i] is not None:
                if is_array(dx[i]):
                    x[-1] = x[-1].reshape(-1, *dx[i])
                else:
                    x[-1] = x[-1].reshape(-1, dx[i])
        elif dx is not None:
            if is_array(dx):
                x[-1] = x[-1].reshape(-1, *dx)
            else:
                x[-1] = x[-1].reshape(-1, dx)

    if len(setup) == 0:
        y = call(f"{inputs[0]} | {target}")
    else:
        y = call(target)
    if dy is not None:
        y = y.reshape(-1, dy)

    module = compose(*modules)
    if len(key) == 0:
        y_hat = module(*x, **opt)
    else:
        x = {k: v for k, v in zip(key, x)}
        y_hat = module(**x, **opt)

    if get is not None:
        if not is_array(get):
            get = [get]
        for g in get:
            y_hat = y_hat[g]
    y_hat = y_hat.cpu().numpy()

    if sr is not None:
        sf.write("output.wav", y_hat / 32768, sr)
        sf.write("target.wav", y / 32768, sr)

    if verbose:
        print(f"Output: {y_hat}")
        print(f"Target: {y}")

    if eq is None:
        assert allclose(y_hat, y, dtype=dtype, **kwargs), (
            f"Output: {y_hat}\nTarget: {y}"
        )
    else:
        assert eq(y_hat, y, **kwargs), f"Output: {y_hat}\nTarget: {y}"

    for cmd in teardown:
        call(cmd, get=False)


def check_confidence(
    device,
    dtype,
    module,
    func,
    size,
):
    x = torch.randn(*size, device=device, dtype=dtype)
    y = func(x.cpu().numpy())
    y_hat = module(x).cpu().numpy()
    assert allclose(y_hat, y, dtype=dtype), f"Output: {y_hat}\nTarget: {y}"


def check_differentiability(
    device,
    dtype,
    modules,
    shapes,
    *,
    complex_input=False,
    checks=None,
    scales=None,
    opt={},
    load=1,
    check_zero_grad=True,
    check_nan_grad=True,
    check_inf_grad=True,
):
    if complex_input:
        dtype = dtype_to_complex_dtype(dtype)

    if not is_array(modules):
        modules = [modules]
    if not is_array(shapes[0]):
        shapes = [shapes]
    if checks is None:
        checks = [True] * len(shapes)

    x = []
    for shape in shapes:
        x.append(torch.randn(*shape, requires_grad=True, device=device, dtype=dtype))
    if scales is None:
        xs = x
    else:
        xs = [scale * x_ for scale, x_ in zip(scales, x)]

    module = compose(*[m.to(device) if hasattr(m, "to") else m for m in modules])
    optimizer = torch.optim.SGD(x, lr=0.01)

    for i in range(load):
        if i == 1:
            s = time.process_time()
        y = module(*xs, **opt)
        optimizer.zero_grad()
        loss = y.mean()
        loss.backward()
        optimizer.step()

    if 1 < load:
        e = time.process_time()
        print(f"time: {e - s}")

    for i in range(len(x)):
        if not checks[i]:
            continue
        g = x[i].grad.cpu().numpy()
        if check_zero_grad and not np.any(g):
            warnings.warn(f"Detected zero gradient at {i}-th input")
        if check_nan_grad and np.any(np.isnan(g)):
            warnings.warn(f"Detected NaN-gradient at {i}-th input")
        if check_inf_grad and np.any(np.isinf(g)):
            warnings.warn(f"Detected Inf-gradient at {i}-th input")


def check_various_shape(module, shapes, *, preprocess=None):
    if is_array(shapes[0][0]):
        xs = [torch.randn(*shape) for shape in shapes[0]]
    else:
        xs = [torch.randn(*shapes[0])]
    if preprocess is not None:
        xs = [preprocess(x) for x in xs]
    for i, shape in enumerate(shapes):
        if is_array(shapes[0][0]):
            x = [x.view(*shape) for x, shape in zip(xs, shape)]
        else:
            x = [xs[0].view(shape)]
        y = module(*x).view(-1)
        if i == 0:
            target = y
        else:
            assert torch.allclose(y, target)


def check_learnable(module, shape, complex_input=False):
    dtype = None
    if complex_input:
        dtype = dtype_to_complex_dtype(dtype)

    params_before = []
    for p in module.parameters():
        params_before.append(p.clone())

    optimizer = torch.optim.SGD(module.parameters(), lr=0.01)
    x = torch.randn(*shape, dtype=dtype)
    y = module(x)
    optimizer.zero_grad()
    loss = y.mean()
    loss.backward()
    optimizer.step()

    params_after = []
    for p in module.parameters():
        params_after.append(p.clone())

    for pb, pa in zip(params_before, params_after):
        assert not torch.allclose(pb, pa)
