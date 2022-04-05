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
@pytest.mark.parametrize("reduction", ["none", "mean"])
def test_compatibility(device, reduction, B=2, L=10):
    if device == "cuda" and not torch.cuda.is_available():
        return

    tmp1 = "rmse.tmp1"
    tmp2 = "rmse.tmp2"
    U.call(f"nrand -s 1 -l {B*L} > {tmp1}", get=False)
    U.call(f"nrand -s 2 -l {B*L} > {tmp2}", get=False)

    rmse = diffsptk.RMSE(reduction=reduction).to(device)
    x1 = torch.from_numpy(U.call(f"cat {tmp1}").reshape(-1, L)).to(device)
    x2 = torch.from_numpy(U.call(f"cat {tmp2}").reshape(-1, L)).to(device)

    opt = "-f" if reduction == "none" else ""
    y = U.call(f"rmse {opt} -l {L} {tmp1} {tmp2}")
    U.call(f"rm {tmp1} {tmp2}", get=False)
    U.check_compatibility(y, rmse, x1, x2)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_differentiable(device, B=2, L=10):
    if device == "cuda" and not torch.cuda.is_available():
        return

    rmse = diffsptk.RMSE().to(device)
    x1 = torch.randn(B, L, requires_grad=True, device=device)
    x2 = torch.randn(B, L, requires_grad=True, device=device)
    U.check_differentiable(rmse, x1, x2)
