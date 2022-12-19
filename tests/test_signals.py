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


@pytest.mark.parametrize("n", [0, 1, 2])
def test_train(n, m=9, p=2.3):
    y = diffsptk.train(m, p, norm=n)
    y_ = U.call(f"train -m {m} -p {p} -n {n}")
    assert U.allclose(y, y_)


def test_nrand(m=10000, u=3, v=4):
    y = torch.var_mean(diffsptk.nrand(m, mean=u, var=v), unbiased=False)
    y_ = U.call(f"nrand -m {m} -u {u} -v {v} | vstat")
    assert U.allclose(y[1], y_[0], rtol=0.1)
    assert U.allclose(y[0], y_[1], rtol=0.1)
