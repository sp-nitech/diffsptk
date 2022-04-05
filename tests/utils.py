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
import torch


def call(cmd, get=True, double=True):
    if get:
        res = subprocess.run(
            cmd + " | x2x +da -f %.12f",
            shell=True,
            text=True,
            stdout=subprocess.PIPE,
        )
        data = np.fromstring(
            res.stdout, sep="\n", dtype=np.float64 if double else np.float32
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


def lap():
    return time.process_time()


def check_compatibility(y_, module, *x, opt={}, verbose=False):
    y = module(*x, **opt).cpu().numpy()
    if verbose:
        print(f"Output: {y}")
        print(f"Target: {y_}")
    assert np.allclose(y, y_)


def check_differentiable(func, *x, opt={}, load=1):
    optimizer = torch.optim.SGD(x, lr=0.001)

    s = lap()
    for _ in range(load):
        y = func(*x, **opt)
        optimizer.zero_grad()
        loss = y.mean()
        loss.backward()
        optimizer.step()
    e = lap()

    if load > 1:
        print(f"time: {e - s}")

    class_name = func.__class__.__name__
    zero_grad_class_names = ("ZeroCrossingAnalysis", "UniformQuantization")
    if not any([class_name == name for name in zero_grad_class_names]):
        for i in range(len(x)):
            g = x[i].grad.cpu().numpy()
            if not np.any(g):
                warnings.warn(f"detect zero-gradient at {i}-th input")
    if True:
        for i in range(len(x)):
            g = x[i].grad.cpu().numpy()
            if np.any(np.isnan(g)):
                warnings.warn(f"detect NaN-gradient at {i}-th input")


def compose(*fs):
    def compose2_outer_kwargs(f, g):
        return lambda *args, **kwargs: f(g(*args), **kwargs)

    return functools.reduce(compose2_outer_kwargs, fs)
