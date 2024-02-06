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
def test_compatibility(device, M=19, N=30, L=256, B=2):
    mpir2c = diffsptk.MinimumPhaseImpulseResponseToCepstrum(M, N, L)

    U.check_compatibility(
        device,
        mpir2c,
        [],
        f"nrand -l {B*L} | fftcep -l {L} -m {M} | c2mpir -m {M} -l {N}",
        f"mpir2c -M {M} -l {N}",
        [],
        dx=N,
        dy=M + 1,
    )

    U.check_differentiability(device, [mpir2c, torch.abs], [B, N])
