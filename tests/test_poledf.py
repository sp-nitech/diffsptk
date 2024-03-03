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

import numpy as np
import pytest

import diffsptk
import tests.utils as U


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("module", [False, True])
@pytest.mark.parametrize("ignore_gain", [False, True])
def test_compatibility(device, module, ignore_gain, M=3, T=100, P=10):
    poledf = U.choice(
        module,
        diffsptk.AllPoleDigitalFilter,
        diffsptk.functional.poledf,
        {"filter_order": M},
        {"frame_period": P, "ignore_gain": ignore_gain},
        n_input=2,
    )

    tmp1 = "poledf.tmp1"
    tmp2 = "poledf.tmp2"
    opt = "-k" if ignore_gain else ""
    U.check_compatibility(
        device,
        poledf,
        [f"nrand -l {T} > {tmp1}", f"nrand -l {T//P*(M+1)} > {tmp2}"],
        [f"cat {tmp1}", f"cat {tmp2}"],
        f"poledf {tmp2} < {tmp1} -m {M} -p {P} {opt}",
        [f"rm {tmp1} {tmp2}"],
        dx=[None, M + 1],
        eq=lambda a, b: np.corrcoef(a, b)[0, 1] > 0.99,
    )

    U.check_differentiability(device, poledf, [(T,), (T // P, M + 1)])
