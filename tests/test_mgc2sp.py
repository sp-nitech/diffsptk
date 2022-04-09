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
@pytest.mark.parametrize("out_format", [0, 1, 2, 3, 4, 5, 6])
def test_compatibility(device, out_format, M=7, L=16):
    if device == "cuda" and not torch.cuda.is_available():
        return

    mgc2sp = diffsptk.MelGeneralizedCepstrumToSpectrum(M, L, out_format=out_format).to(
        device
    )
    x = torch.from_numpy(U.call(f"step -v 0.1 -m {M}").reshape(-1, M + 1)).to(device)
    y = U.call(f"step -v 0.1 -m {M} | mgc2sp -m {M} -l {L} -o {out_format}").reshape(
        -1, L // 2 + 1
    )
    U.check_compatibility(y, mgc2sp, x)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_differentiable(device, M=7, L=16, B=2):
    if device == "cuda" and not torch.cuda.is_available():
        return

    mgc2sp = diffsptk.MelGeneralizedCepstrumToSpectrum(M, L, out_format=1).to(device)
    x = torch.ones(B, M + 1, requires_grad=True, device=device)
    U.check_differentiable(mgc2sp, x)
