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
@pytest.mark.parametrize("P", [1, 4])
def test_compatibility(device, module, P, N=10):
    linear_intpl = U.choice(
        module,
        diffsptk.LinearInterpolation,
        diffsptk.functional.linear_intpl,
        {},
        {"upsampling_factor": P},
    )

    tmp = "linear_intpl.tmp"
    U.check_compatibility(
        device,
        linear_intpl,
        [f"ramp -s 1 -e {N} > {tmp}"],
        f"cat {tmp}",
        (
            f"cat {tmp}"
            if P == 1
            else f"step -v 1 -l {N*P} | zerodf {tmp} -i 1 -m 0 -p {P}"
        ),
        [f"rm {tmp}"],
    )

    U.check_differentiability(device, linear_intpl, [N])


def test_various_shape(P=4, N=10):
    linear_intpl = diffsptk.LinearInterpolation(P)
    U.check_various_shape(linear_intpl, [(N,), (N, 1), (1, N, 1)])
