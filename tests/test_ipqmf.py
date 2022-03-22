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
@pytest.mark.parametrize("M", [10, 11])
def test_compatibility(device, M, K=4, L=20):
    if device == "cuda" and not torch.cuda.is_available():
        return

    ipqmf = diffsptk.IPQMF(K, M).to(device)
    x = torch.from_numpy(U.call(f"nrand -l {K*L}").reshape(L, K)).t()
    x = x.unsqueeze(0).to(device)
    y = U.call(f"nrand -l {K*L} | ipqmf -k {K} -m {M}").reshape(1, 1, -1)
    U.check_compatibility(y, ipqmf, x)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_differentiable(device, K=4, M=8, B=2, L=20):
    if device == "cuda" and not torch.cuda.is_available():
        return

    ipqmf = diffsptk.IPQMF(K, M).to(device)
    x = torch.randn(B, K, L, requires_grad=True, device=device)
    U.check_differentiable(ipqmf, x)
