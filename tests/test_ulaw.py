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
def test_compatibility(device, v=10, u=255, L=10):
    if device == "cuda" and not torch.cuda.is_available():
        return

    ulaw = diffsptk.MuLawCompression(v, u).to(device)
    x = torch.arange(L).to(device)
    y = U.call(f"ramp -l {L} | ulaw -v {v} -u {u}")
    U.check_compatibility(y, ulaw, x)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_differentiable(device, L=20):
    if device == "cuda" and not torch.cuda.is_available():
        return

    ulaw = diffsptk.MuLawCompression().to(device)
    x = torch.randn(L, requires_grad=True, device=device)
    U.check_differentiable(ulaw, x)
