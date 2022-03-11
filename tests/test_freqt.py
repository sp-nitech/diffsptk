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


def test_compatibility(m=19, M=29, alpha=0.1, B=2):
    freqt = diffsptk.FrequencyTransform(m, M, alpha)
    x = torch.from_numpy(call(f"nrand -l {B*(m+1)}").reshape(-1, m + 1))
    y = freqt(x).cpu().numpy()

    y_ = call(f"nrand -l {B*(m+1)} | freqt -m {m} -M {M} -A {alpha}").reshape(-1, M + 1)
    assert np.allclose(y, y_)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_differentiable(device, m=19, M=29, alpha=0.1, B=2):
    if device == "cuda" and not torch.cuda.is_available():
        return

    freqt = diffsptk.FrequencyTransform(m, M, alpha).to(device)
    x = torch.randn(B, m + 1, requires_grad=True, device=device)
    y = freqt(x)

    optimizer = torch.optim.SGD([x], lr=0.001)
    optimizer.zero_grad()
    loss = y.mean()
    loss.backward()
    optimizer.step()
