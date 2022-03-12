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

import numpy as np
import pytest
import torch

import diffsptk
from tests.utils import call


@pytest.mark.parametrize("norm", [0, 1, 2])
@pytest.mark.parametrize("win", [0, 1, 2, 3, 4, 5])
def test_compatibility(norm, win, L1=10, L2=12):
    window = diffsptk.Window(L1, L2, norm=norm, window=win)
    x = torch.ones(L1, dtype=torch.float64).view(1, L1)
    y = window(x).cpu().numpy()

    y_ = call(f"step -l {L1} | window -n {norm} -w {win} -l {L1} -L {L2}").reshape(
        -1, L2
    )
    assert np.allclose(y, y_)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_differentiable(device, L=10, B=2):
    if device == "cuda" and not torch.cuda.is_available():
        return

    window = diffsptk.Window(L).to(device)
    x = torch.randn(B, L, requires_grad=True, device=device)
    y = window(x)

    optimizer = torch.optim.SGD([x], lr=0.001)
    optimizer.zero_grad()
    loss = y.mean()
    loss.backward()
    optimizer.step()
