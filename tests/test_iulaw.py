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


def test_compatibility(v=10, u=255, L=10):
    iulaw = diffsptk.MuLawExpansion(v, u)
    x = torch.arange(L, dtype=torch.float32)
    y = iulaw(x).cpu().numpy()

    y_ = call(f"ramp -l {L} | iulaw -v {v} -u {u}")
    assert np.allclose(y, y_)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_differentiable(device, L=20):
    if device == "cuda" and not torch.cuda.is_available():
        return

    iulaw = diffsptk.MuLawExpansion().to(device)
    x = torch.randn(L, requires_grad=True, device=device)
    check(iulaw, x)
