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
def test_compatibility(device, dtype, module, M=9, B=2):
    par2is = U.choice(
        module,
        diffsptk.ParcorCoefficientsToInverseSine,
        diffsptk.functional.par2is,
        {"par_order": M},
    )

    U.check_compatibility(
        device,
        dtype,
        par2is,
        [],
        f"nrand -l {B * (M + 1)} -v 0.1",
        "sopr -ASIN -m 2 -d pi",
        [],
        dx=M + 1,
        dy=M + 1,
        eq=lambda x, y: U.allclose(x[:, 1:], y[:, 1:]),
    )

    U.check_differentiability(device, dtype, par2is, [B, M + 1])
