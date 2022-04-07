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
def test_compatibility(device, M=9, gamma=0.9, B=2):
    if device == "cuda" and not torch.cuda.is_available():
        return

    lpc2par = diffsptk.LinearPredictiveCoefficientsToParcorCoefficients(
        M, gamma=gamma, warn_type="ignore"
    ).to(device)
    x = torch.from_numpy(U.call(f"nrand -l {B*(M+1)}").reshape(-1, M + 1)).to(device)
    y = U.call(f"nrand -l {B*(M+1)} | lpc2par -m {M} -g {gamma}").reshape(-1, M + 1)
    U.check_compatibility(y, lpc2par, x)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_differentiable(device, M=9, B=2):
    if device == "cuda" and not torch.cuda.is_available():
        return

    lpc2par = diffsptk.LinearPredictiveCoefficientsToParcorCoefficients(
        M, warn_type="ignore"
    ).to(device)
    x = torch.randn(B, M + 1, requires_grad=True, device=device)
    U.check_differentiable(lpc2par, x)
