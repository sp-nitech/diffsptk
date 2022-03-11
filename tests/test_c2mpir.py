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


def test_compatibility(M=19, N=30, B=2):
    c2ir = diffsptk.CepstrumToImpulseResponse(M, N)
    x = torch.from_numpy(call(f"nrand -l {B*(M+1)}").reshape(-1, M + 1))
    y = c2ir(x).cpu().numpy()

    y_ = call(f"nrand -l {B*(M+1)} | c2mpir -m {M} -l {N}").reshape(-1, N)
    assert np.allclose(y, y_)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_differentiable(device, M=19, N=30, B=2):
    c2ir = diffsptk.CepstrumToImpulseResponse(M, N).to(device)
    x = torch.randn(B, M + 1, requires_grad=True, device=device)
    y = c2ir(x)

    optimizer = torch.optim.SGD([x], lr=0.001)
    optimizer.zero_grad()
    loss = y.mean()
    loss.backward()
    optimizer.step()
