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
from tests.utils import check


@pytest.mark.parametrize("fl", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("fp", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("center", [True, False])
def test_compatibility(fl, fp, center, T=20):
    frame = diffsptk.Frame(fl, fp, center=center)
    x = torch.arange(T, dtype=torch.float32).view(1, -1)
    y = frame(x).cpu().numpy()

    n = 0 if center else 1
    y_ = call(f"ramp -l {T} | frame -l {fl} -p {fp} -n {n}").reshape(-1, fl)
    assert np.allclose(y, y_)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_differentiable(device, fl=5, fp=3, B=4, T=20):
    if device == "cuda" and not torch.cuda.is_available():
        return

    frame = diffsptk.Frame(fl, fp).to(device)
    x = torch.randn(B, T, requires_grad=True, device=device)
    check(frame, x)
