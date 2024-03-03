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
def test_compatibility(device, module, L=32, M=9, gamma=0.9, B=2):
    lpc2par = U.choice(
        module,
        diffsptk.LinearPredictiveCoefficientsToParcorCoefficients,
        diffsptk.functional.lpc2par,
        {"lpc_order": M},
        {"gamma": gamma},
    )

    U.check_compatibility(
        device,
        lpc2par,
        [],
        f"nrand -l {B*L} | lpc -l {L} -m {M}",
        f"lpc2par -m {M} -g {gamma}",
        [],
        dx=M + 1,
        dy=M + 1,
    )

    U.check_differentiability(device, lpc2par, [B, M + 1])
