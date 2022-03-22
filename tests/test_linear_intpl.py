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
def test_compatibility(device, P=4, N=10):
    if device == "cuda" and not torch.cuda.is_available():
        return

    tmp = "linear_intpl.tmp"
    U.call(f"ramp -s 1 -e {N} > {tmp}", get=False)

    linear_intpl = diffsptk.LinearInterpolation(P).to(device)
    x = torch.from_numpy(U.call(f"cat {tmp}").reshape(1, -1, 1)).to(device)
    y = U.call(f"step -v 1 -l {N*P} | zerodf {tmp} -i 1 -m 0 -p {P}").reshape(-1, 1)

    U.call(f"rm {tmp}", get=False)
    U.check_compatibility(y, linear_intpl, x)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_differentiable(device, P=4, B=2, N=10, D=2):
    if device == "cuda" and not torch.cuda.is_available():
        return

    linear_intpl = diffsptk.LinearInterpolation(P).to(device)
    x = torch.randn(B, N, D, requires_grad=True, device=device)
    U.check_differentiable(linear_intpl, x)
