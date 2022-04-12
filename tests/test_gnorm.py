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
@pytest.mark.parametrize("gamma", [0, 1])
def test_compatibility(device, gamma, B=2, M=4):
    if device == "cuda" and not torch.cuda.is_available():
        return

    gnorm = diffsptk.GeneralizedCepstrumGainNormalization(M, gamma).to(device)
    x = torch.from_numpy(U.call(f"nrand -l {B*(M+1)}").reshape(-1, M + 1)).to(device)
    y = U.call(f"nrand -l {B*(M+1)} | gnorm -g {gamma} -m {M}").reshape(-1, M + 1)
    U.check_compatibility(y, gnorm, x)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_compatibility2(device, c=2, B=2, M=4):
    if device == "cuda" and not torch.cuda.is_available():
        return

    gnorm = diffsptk.GeneralizedCepstrumGainNormalization(M, c=c).to(device)
    x = torch.from_numpy(U.call(f"nrand -l {B*(M+1)}").reshape(-1, M + 1)).to(device)
    y = U.call(f"nrand -l {B*(M+1)} | gnorm -c {c} -m {M}").reshape(-1, M + 1)
    U.check_compatibility(y, gnorm, x)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_differentiable(device, gamma=1, B=2, M=4):
    if device == "cuda" and not torch.cuda.is_available():
        return

    gnorm = diffsptk.GeneralizedCepstrumGainNormalization(M, gamma).to(device)
    x = torch.randn(B, M + 1, requires_grad=True, device=device)
    U.check_differentiable(gnorm, x)
