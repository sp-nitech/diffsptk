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
@pytest.mark.parametrize("reduction", ["none", "batchmean", "mean", "sum"])
def test_compatibility(device, dtype, module, reduction, B=2, M=19):
    cdist = U.choice(
        module,
        diffsptk.CepstralDistance,
        diffsptk.functional.cdist,
        {"full": True, "reduction": reduction},
    )

    opt = "-f" if reduction == "none" else ""
    if reduction == "none" or reduction == "batchmean":
        mul = 1
    elif reduction == "mean":
        mul = M**-0.5
    elif reduction == "sum":
        mul = B
    else:
        raise ValueError

    tmp1 = "cdist.tmp1"
    tmp2 = "cdist.tmp2"
    U.check_compatibility(
        device,
        dtype,
        cdist,
        [
            f"nrand -s 1 -l {B * (M + 1)} > {tmp1}",
            f"nrand -s 2 -l {B * (M + 1)} > {tmp2}",
        ],
        [f"cat {tmp1}", f"cat {tmp2}"],
        f"cdist -m {M} {opt} {tmp1} {tmp2} | sopr -m {mul}",
        [f"rm {tmp1} {tmp2}"],
        dx=M + 1,
    )

    U.check_differentiability(device, dtype, cdist, [(B, M + 1), (B, M + 1)])
