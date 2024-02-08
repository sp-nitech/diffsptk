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
def test_compatibility(device, module, M=8, L=16, B=2):
    ndps2c = U.choice(
        module,
        diffsptk.NegativeDerivativeOfPhaseSpectrumToCepstrum,
        diffsptk.functional.ndps2c,
        {"fft_length": L},
        {"cep_order": M},
    )

    H = L // 2 + 1
    U.check_compatibility(
        device,
        ndps2c,
        [],
        f"nrand -l {B*H}",
        f"ndps2c -m {M} -l {L}",
        [],
        dx=H,
        dy=M + 1,
    )

    U.check_differentiability(device, ndps2c, [B, H])
