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
def test_compatibility(device, P=2, S=1, T=20, L=4):
    if device == "cuda" and not torch.cuda.is_available():
        return

    decimate = diffsptk.Decimation(P, S).to(device)
    x = torch.arange(T * L).view(T, L).to(device)
    y = U.call(f"ramp -l {T*L} | decimate -l {L} -p {P} -s {S}").reshape(-1, L)
    U.check_compatibility(y, decimate, x)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_differentiable(device, P=3, S=1, B=2, T=20):
    if device == "cuda" and not torch.cuda.is_available():
        return

    decimate = diffsptk.Decimation(P, S).to(device)
    x = torch.randn(B, T, requires_grad=True, device=device)
    U.check_differentiable(decimate, x, opt={"dim": -1})
