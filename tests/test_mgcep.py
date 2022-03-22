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
@pytest.mark.parametrize("n_iter", [0, 3])
def test_compatibility(device, n_iter, M=8, L=16, B=2, alpha=0.1):
    if device == "cuda" and not torch.cuda.is_available():
        return

    spec = diffsptk.Spectrum(L, eps=0).to(device)
    mcep = diffsptk.MelCepstralAnalysis(M, L, alpha, n_iter=n_iter).to(device)
    x = spec(torch.from_numpy(U.call(f"nrand -l {B*L}").reshape(-1, L)).to(device))
    y = U.call(
        f"nrand -l {B*L} | mgcep -g 0 -d 0 -i {n_iter} -l {L} -m {M} -a {alpha}"
    ).reshape(-1, M + 1)
    U.check_compatibility(y, mcep, x)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_differentiable(device, n_iter=3, M=4, L=16, B=2, alpha=0.1):
    if device == "cuda" and not torch.cuda.is_available():
        return

    spec = diffsptk.Spectrum(L).to(device)
    mcep = diffsptk.MelCepstralAnalysis(M, L, alpha, n_iter=n_iter).to(device)
    x = torch.randn(B, L, requires_grad=True, device=device)
    U.check_differentiable(U.compose(mcep, spec), x)
