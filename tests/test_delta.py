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
def test_compatibility(device, coef=[[-1, 1], [-1, 0, 1]], T=10, L=2):
    if device == "cuda" and not torch.cuda.is_available():
        return

    delta = diffsptk.Delta(coef).to(device)
    x = torch.from_numpy(U.call(f"nrand -l {T*L}").reshape(1, -1, L)).to(device)
    d = " ".join(["-d " + " ".join([str(x) for x in c]) for c in coef])
    y = U.call(f"nrand -l {T*L} | delta -l {L} {d}").reshape(-1, L * (1 + len(coef)))
    U.check_compatibility(y, delta, x)


def test_compatibility2(device="cpu", width=[2, 3], T=10, L=2):
    delta = diffsptk.Delta(regression_width=width).to(device)
    x = torch.from_numpy(U.call(f"nrand -l {T*L}").reshape(1, -1, L)).to(device)
    r = " ".join([str(x) for x in width])
    y = U.call(f"nrand -l {T*L} | delta -l {L} -r {r}").reshape(
        -1, L * (1 + len(width))
    )
    U.check_compatibility(y, delta, x)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_differentiable(device, coef=[1, 1, 1], B=2, T=10, L=2):
    if device == "cuda" and not torch.cuda.is_available():
        return

    delta = diffsptk.Delta(coef).to(device)
    x = torch.randn(B, T, L, requires_grad=True, device=device)
    U.check_differentiable(delta, x)
