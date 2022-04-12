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
@pytest.mark.parametrize("o", [0, 1, 2, 3])
def test_compatibility(device, o, M=3, L=14, B=2):
    if device == "cuda" and not torch.cuda.is_available():
        return

    if o == 0:
        opt = {"norm": False}
    elif o == 1:
        opt = {"norm": True}
    elif o == 2:
        opt = {"acf": "biased"}
    elif o == 3:
        opt = {"acf": "unbiased"}
    acorr = diffsptk.AutocorrelationAnalysis(M, L, **opt).to(device)
    x = torch.from_numpy(U.call(f"nrand -l {B*L}").reshape(-1, L)).to(device)
    y = U.call(f"nrand -l {B*L} | acorr -l {L} -m {M} -o {o}").reshape(-1, M + 1)
    U.check_compatibility(y, acorr, x)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_differentiable(device, M=3, L=14, B=2):
    if device == "cuda" and not torch.cuda.is_available():
        return

    acorr = diffsptk.AutocorrelationAnalysis(M, L).to(device)
    x = torch.randn(B, L, requires_grad=True, device=device)
    U.check_differentiable(acorr, x)
