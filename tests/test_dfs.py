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


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("b", [(1,), (-0.42, 1)])
@pytest.mark.parametrize("a", [(1,), (1, -0.42)])
def test_compatibility(device, b, a, T=100):
    dfs = diffsptk.IIR(b, a, ir_length=30)

    bb = " ".join([str(x) for x in b])
    aa = " ".join([str(x) for x in a])

    U.check_compatibility(
        device,
        dfs,
        [],
        f"nrand -l {T}",
        f"dfs -b {bb} -a {aa}",
        [],
    )

    U.check_differentiability(device, dfs, [T])


def test_compatibility_func(b=[-0.42, 1], a=[1, -0.42], T=20):
    x = diffsptk.nrand(T)

    y1 = diffsptk.IIR(b=b)(x)
    y2 = diffsptk.functional.dfs(x, b=torch.tensor(b))
    assert U.allclose(y1, y2)

    y1 = diffsptk.IIR(a=a, ir_length=20)(x)
    y2 = diffsptk.functional.dfs(x, a=torch.tensor(a))
    assert U.allclose(y1, y2)


def test_learnable(b=[-0.42, 1], T=20):
    dfs = diffsptk.IIR(b, learnable=True)
    U.check_learnable(dfs, (T,))
