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


@pytest.mark.parametrize("module", [False, True])
@pytest.mark.parametrize("norm", [False, True])
def test_compatibility(device, dtype, module, norm, L=10, T=50):
    zcross = U.choice(
        module,
        diffsptk.ZeroCrossingAnalysis,
        diffsptk.functional.zcross,
        {"frame_length": L, "norm": norm, "softness": 1e-3},
    )

    opt = "-o 1" if norm else ""
    U.check_compatibility(
        device,
        dtype,
        zcross,
        [],
        f"nrand -l {T}",
        f"zcross -l {L} {opt}",
        [],
    )


def test_differentiability(device, dtype, L=10, T=50):
    zcross = diffsptk.ZeroCrossingAnalysis(L, softness=1e-1)
    U.check_differentiability(device, dtype, zcross, [T])
