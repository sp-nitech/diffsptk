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
def test_compatibility(device, module, v=10, a=80, L=10):
    ialaw = U.choice(
        module,
        diffsptk.ALawExpansion,
        diffsptk.functional.ialaw,
        {},
        {"abs_max": v, "a": a},
    )

    U.check_compatibility(
        device,
        ialaw,
        [],
        f"ramp -l {L}",
        f"ialaw -v {v} -a {a}",
        [],
    )

    U.check_differentiability(device, ialaw, [L])
