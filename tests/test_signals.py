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

import pytest
import torch

import diffsptk
import tests.utils as U


def test_impulse(m=5):
    y = diffsptk.impulse(m)
    y_ = U.call(f"impulse -m {m}")
    assert U.allclose(y, y_)


def test_step(m=5, v=-1):
    y = diffsptk.step(m, v)
    y_ = U.call(f"step -m {m} -v {v}")
    assert U.allclose(y, y_)


def test_ramp(s=5, e=3, t=-1):
    y = diffsptk.ramp(s, e, t)
    y_ = U.call(f"ramp -s {s} -e {e} -t {t}")
    assert U.allclose(y, y_)

    y = diffsptk.ramp(s)
    y_ = U.call(f"ramp -m {s}")
    assert U.allclose(y, y_)


def test_sin(m=5, p=4, a=-1):
    y = diffsptk.sin(m, p, a)
    y_ = U.call(f"sin -m {m} -p {p} -a {a}")
    assert U.allclose(y, y_)
    y = diffsptk.sin(m)
    y_ = U.call(f"sin -m {m} -p {m + 1}")
    assert U.allclose(y, y_)


@pytest.mark.parametrize("n", [0, 1, 2])
def test_train(n, m=9, p=2.3):
    y = diffsptk.train(m, p, norm=n)
    y_ = U.call(f"train -m {m} -p {p} -n {n}")
    assert U.allclose(y, y_)


@pytest.mark.parametrize("tuple_input", [False, True])
def test_nrand(tuple_input, m=10000, u=3, v=4):
    x = diffsptk.nrand((m,) if tuple_input else m, mean=u, var=v)
    y = torch.var_mean(x, unbiased=False)
    y_ = U.call(f"nrand -m {m} -u {u} -v {v} | vstat")
    assert U.allclose(y[1], y_[0], rtol=0.1)
    assert U.allclose(y[0], y_[1], rtol=0.1)


@pytest.mark.parametrize("tuple_input", [False, True])
def test_rand(tuple_input, m=10000, a=-1, b=1):
    x = diffsptk.rand((m,) if tuple_input else m, a=a, b=b)
    y = torch.stack((torch.max(x), torch.min(x)))
    y_ = U.call(f"rand -m {m} -a {a} -b {b} | minmax")
    assert U.allclose(y[1], y_[0], rtol=0.1)
    assert U.allclose(y[0], y_[1], rtol=0.1)
