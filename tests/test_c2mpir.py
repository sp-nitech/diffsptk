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

import diffsptk
import tests.utils as U


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("module", [False, True])
def test_compatibility(device, module, M=19, N=30, L=512, B=2):
    c2mpir = U.choice(
        module,
        diffsptk.CepstrumToMinimumPhaseImpulseResponse,
        diffsptk.functional.c2mpir,
        {"cep_order": M},
        {"ir_length": N, "n_fft": L},
    )

    U.check_compatibility(
        device,
        c2mpir,
        [],
        f"nrand -l {B*L} | fftcep -l {L} -m {M}",
        f"c2mpir -m {M} -l {N}",
        [],
        dx=M + 1,
        dy=N,
    )

    U.check_differentiability(device, c2mpir, [B, M + 1])
