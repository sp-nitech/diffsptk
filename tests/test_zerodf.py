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


def test_compatibility(M=3, T=100, P=10):
    tmp1 = "zerodf.tmp1"
    tmp2 = "zerodf.tmp2"
    call(f"nrand -l {T} > {tmp1}", get=False)
    call(f"nrand -l {T//P*(M+1)} > {tmp2}", get=False)

    zerodf = diffsptk.AllZeroDigitalFilter(M, P)
    x = torch.from_numpy(call(f"cat {tmp1}").reshape(1, -1))
    h = torch.from_numpy(call(f"cat {tmp2}").reshape(1, -1, M + 1))
    y = zerodf(x, h).cpu().numpy()

    y_ = call(f"zerodf {tmp2} < {tmp1} -i 1 -m {M} -p {P}").reshape(1, -1)
    call(f"rm {tmp1} {tmp2}", get=False)
    assert np.allclose(y, y_, atol=1e-6)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_differentiable(device, B=4, T=20, D=2):
    if device == "cuda" and not torch.cuda.is_available():
        return

    zerodf = diffsptk.AllZeroDigitalFilter(D - 1).to(device)
    x = torch.randn(B, T, requires_grad=True, device=device)
    h = torch.randn(B, T, D, requires_grad=True, device=device)
    check(zerodf, x, h)
