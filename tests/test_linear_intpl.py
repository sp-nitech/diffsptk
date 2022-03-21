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


def test_compatibility(P=4, N=10):
    tmp = "linear_intpl.tmp"
    call(f"ramp -s 1 -e {N} > {tmp}", get=False)

    linear_intpl = diffsptk.LinearInterpolation(P)
    x = torch.from_numpy(call(f"cat {tmp}").reshape(1, -1, 1))
    y = linear_intpl(x).cpu().numpy().reshape(-1)

    y_ = call(f"step -v 1 -l {N*P} | zerodf {tmp} -i 1 -m 0 -p {P}")
    call(f"rm {tmp}", get=False)
    assert np.allclose(y, y_)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_differentiable(device, P=4, B=2, N=10, D=2):
    if device == "cuda" and not torch.cuda.is_available():
        return

    linear_intpl = diffsptk.LinearInterpolation(P).to(device)
    x = torch.randn(B, N, D, requires_grad=True, device=device)
    check(linear_intpl, x)
