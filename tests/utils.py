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

import numpy as np
import soundfile as sf
import torch


def is_array(x):
    return type(x) is list or type(x) is tuple


def compose(*fs):
    def compose2_outer_kwargs(f, g):
        return lambda *args, **kwargs: f(g(*args), **kwargs)

    return functools.reduce(compose2_outer_kwargs, fs)


def choice(first, a, b, a_params, params={}, n_input=1):
    if first:
        return a(**a_params, **params)
    elif n_input == 1:

        def func(x, **kwargs):
            return b(x, **params, **kwargs)

        return func
    elif n_input == 2:

        def func(x, y, **kwargs):
            return b(x, y, **params, **kwargs)

        return func
    raise ValueError("n_input must be 1 or 2.")


def allclose(a, b, rtol=None, atol=None):
    is_double = torch.get_default_dtype() == torch.double
    if rtol is None:
        rtol = 1e-5 if is_double else 1e-4
    if atol is None:
        atol = 1e-8 if is_double else 1e-6
    return np.allclose(a, b, rtol=rtol, atol=atol)


def call(cmd, get=True):
    if get:
        res = subprocess.run(
            cmd + " | x2x +da -f %.15g",
            shell=True,
            text=True,
            stdout=subprocess.PIPE,
        )
        is_double = torch.get_default_dtype() == torch.double
        data = np.fromstring(
            res.stdout, sep="\n", dtype=np.double if is_double else np.float32
        )
        assert len(data) > 0, f"Failed to run command {cmd}"
        return data
    else:
        res = subprocess.run(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
        )
        return None


def check_compatibility(
    device,
    modules,
    setup,
    inputs,
    target,
    teardown,
    dx=None,
    dy=None,
    eq=None,
    opt={},
    key=[],
    sr=None,
    verbose=False,
    **kwargs,
):
    if device == "cuda" and not torch.cuda.is_available():
        return

    for cmd in setup:
        call(cmd, get=False)

    if not is_array(modules):
        modules = [modules]
    if not is_array(inputs):
        inputs = [inputs]

    x = []
    for i, cmd in enumerate(inputs):
        x.append(torch.from_numpy(call(cmd)).to(device))
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

    for cmd in teardown:
        call(cmd, get=False)

    module = compose(*[m.to(device) if hasattr(m, "to") else m for m in modules])
    if len(key) == 0:
        y_hat = module(*x, **opt).cpu().numpy()
    else:
        x = {k: v for k, v in zip(key, x)}
        y_hat = module(**x, **opt).cpu().numpy()

    if sr is not None:
        sf.write("output.wav", y_hat / 32768, sr)
        sf.write("target.wav", y / 32768, sr)

    if verbose:
        print(f"Output: {y_hat}")
        print(f"Target: {y}")

    if eq is None:
        assert allclose(y_hat, y, **kwargs), f"Output: {y_hat}\nTarget: {y}"
    else:
        assert eq(y_hat, y, **kwargs), f"Output: {y_hat}\nTarget: {y}"


def check_differentiability(
    device,
    modules,
    shapes,
    *,
    dtype=None,
    checks=None,
    opt={},
    load=1,
    check_zero_grad=True,
    check_nan_grad=True,
    check_inf_grad=True,
):
    if device == "cuda" and not torch.cuda.is_available():
        return

    if not is_array(modules):
        modules = [modules]
    if not is_array(shapes[0]):
        shapes = [shapes]
    if checks is None:
        checks = [True] * len(shapes)

    x = []
    for shape in shapes:
        x.append(torch.randn(*shape, requires_grad=True, device=device, dtype=dtype))

    module = compose(*[m.to(device) if hasattr(m, "to") else m for m in modules])
    optimizer = torch.optim.SGD(x, lr=0.01)

    s = time.process_time()
    for _ in range(load):
        y = module(*x, **opt)
        optimizer.zero_grad()
        loss = y.mean()
        loss.backward()
        optimizer.step()
    e = time.process_time()

    if load > 1:
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
    x = torch.randn(*shapes[0])
    if preprocess is not None:
        x = preprocess(x)
    for i, shape in enumerate(shapes):
        x = x.view(shape)
        y = module(x).view(-1)
        if i == 0:
            target = y
        else:
            assert torch.allclose(y, target)


def check_learnable(module, shape):
    params_before = []
    for p in module.parameters():
        params_before.append(p.clone())

    optimizer = torch.optim.SGD(module.parameters(), lr=0.01)
    x = torch.randn(*shape)
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
