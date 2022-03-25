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
def test_compatibility(device, M=4, L=16, B=2):
    if device == "cuda" and not torch.cuda.is_available():
        return

    c2acr = diffsptk.CepstrumToAutocorrelation(M, L).to(device)
    x = torch.from_numpy(U.call(f"nrand -l {B*(M+1)}").reshape(-1, M + 1)).to(device)
    y = U.call(f"nrand -l {B*(M+1)} | c2acr -m {M} -M {M} -l {L}").reshape(-1, M + 1)
    U.check_compatibility(y, c2acr, x)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_differentiable(device, M=4, L=16, B=2):
    if device == "cuda" and not torch.cuda.is_available():
        return

    c2acr = diffsptk.CepstrumToAutocorrelation(M, L).to(device)
    x = torch.randn(B, M + 1, requires_grad=True, device=device)
    U.check_differentiable(c2acr, x)
