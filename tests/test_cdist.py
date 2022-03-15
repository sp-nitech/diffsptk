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


@pytest.mark.parametrize("reduction", ["none", "batchmean"])
def test_compatibility(reduction, M=19, B=2):
    tmp1 = "cdist.tmp1"
    tmp2 = "cdist.tmp2"
    call(f"nrand -s 1 -l {B*(M+1)} > {tmp1}", get=False)
    call(f"nrand -s 2 -l {B*(M+1)} > {tmp2}", get=False)

    cdist = diffsptk.CepstralDistance(full=True, reduction=reduction)
    x1 = torch.from_numpy(call(f"cat {tmp1}").reshape(-1, M + 1))
    x2 = torch.from_numpy(call(f"cat {tmp2}").reshape(-1, M + 1))
    y = cdist(x1, x2).cpu().numpy()

    opt = "-f" if reduction == "none" else ""
    y_ = call(f"cdist {opt} -m {M} {tmp1} {tmp2}").reshape(-1)
    call(f"rm {tmp1} {tmp2}", get=False)
    assert np.allclose(y, y_)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_differentiable(device, M=19, B=2):
    if device == "cuda" and not torch.cuda.is_available():
        return

    cdist = diffsptk.CepstralDistance().to(device)
    x1 = torch.randn(B, M + 1, requires_grad=True, device=device)
    x2 = torch.randn(B, M + 1, requires_grad=False, device=device)
    check(cdist, x1, x2)
