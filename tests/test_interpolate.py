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


def test_compatibility(p=2, s=1, T=20, L=4):
    interpolate = diffsptk.Interpolation(p, s)
    x = torch.arange(T * L, dtype=torch.float32).view(T, L)
    y = interpolate(x, dim=0).cpu().numpy()

    y_ = call(f"ramp -l {T*L} | interpolate -l {L} -p {p} -s {s}").reshape(-1, L)
    assert np.allclose(y, y_)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_differentiable(device, p=3, B=2, T=20):
    if device == "cuda" and not torch.cuda.is_available():
        return

    interpolate = diffsptk.Interpolation(p).to(device)
    x = torch.randn(B, T, requires_grad=True, device=device)
    y = interpolate(x, dim=-1)

    optimizer = torch.optim.SGD([x], lr=0.001)
    optimizer.zero_grad()
    loss = y.mean()
    loss.backward()
    optimizer.step()
