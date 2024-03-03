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
def test_compatibility(device, module, M=9, B=2):
    par2lar = U.choice(
        module,
        diffsptk.ParcorCoefficientsToLogAreaRatio,
        diffsptk.functional.par2lar,
        {"par_order": M},
    )

    U.check_compatibility(
        device,
        par2lar,
        [],
        f"nrand -l {B*(M+1)} -v 0.1",
        f"par2lar -m {M}",
        [],
        dx=M + 1,
        dy=M + 1,
    )

    U.check_differentiability(device, par2lar, [B, M + 1])
