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
@pytest.mark.parametrize("norm", [0, 1, 2])
@pytest.mark.parametrize("win", [0, 1, 2, 3, 4, 5])
def test_compatibility(device, norm, win, L1=10, L2=12):
    if device == "cuda" and not torch.cuda.is_available():
        return

    window = diffsptk.Window(L1, L2, norm=norm, window=win).to(device)
    x = torch.ones(L1).view(1, L1).to(device)
    y = U.call(f"step -l {L1} | window -n {norm} -w {win} -l {L1} -L {L2}").reshape(
        -1, L2
    )
    U.check_compatibility(y, window, x)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_differentiable(device, L=10, B=2):
    if device == "cuda" and not torch.cuda.is_available():
        return

    window = diffsptk.Window(L).to(device)
    x = torch.randn(B, L, requires_grad=True, device=device)
    U.check_differentiable(window, x)
