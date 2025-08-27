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
@pytest.mark.parametrize("metric", [0, 1, 2, 3])
@pytest.mark.parametrize("p", [0, 1, 2, 3, 4, 5, 6])
def test_compatibility(device, dtype, module, metric, p, M=0, T1=10, T2=10):
    dtw = U.choice(
        module,
        diffsptk.DynamicTimeWarping,
        diffsptk.functional.dtw,
        {"metric": metric, "p": p, "softness": 1e-3},
    )

    sopr = "| sopr -ABS" if metric == 3 else ""
    tmp1 = "dtw.tmp1"
    tmp2 = "dtw.tmp2"
    tmp3 = "dtw.tmp3"
    U.check_compatibility(
        device,
        dtype,
        dtw,
        [
            f"nrand -s 1 -l {T1 * (M + 1)} {sopr} > {tmp1}",
            f"nrand -s 2 -l {T2 * (M + 1)} {sopr} > {tmp2}",
        ],
        [f"cat {tmp1}", f"cat {tmp2}"],
        (
            f"dtw -m {M} -d {metric} -p {p} < {tmp1} {tmp2} -S {tmp3} > /dev/null "
            f"&& cat {tmp3}"
        ),
        [f"rm {tmp1} {tmp2} {tmp3}"],
        dx=M + 1,
    )

    U.check_compatibility(
        device,
        dtype,
        dtw,
        [
            f"nrand -s 1 -l {T1} {sopr} > {tmp1}",
            f"nrand -s 2 -l {T2} {sopr} > {tmp2}",
        ],
        [f"cat {tmp1}", f"cat {tmp2}"],
        (
            f"dtw -m 0 -d {metric} -p {p} < {tmp1} {tmp2} -P {tmp3} > /dev/null "
            f"&& x2x +id {tmp3}"
        ),
        [f"rm {tmp1} {tmp2} {tmp3}"],
        dy=2,
        opt={"return_indices": True},
        get=[1, 0],
    )

    U.check_differentiability(device, dtype, dtw, [(T1, M + 1), (T2, M + 1)])
