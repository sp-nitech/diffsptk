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
@pytest.mark.parametrize("P", [4])
@pytest.mark.parametrize("strict", [False])
@pytest.mark.parametrize("mod_type", ["fast"])
def test_compatibility(device, P, strict, mod_type, M=9, L=32, alpha=0.1, B=2):
    mlsacheck = diffsptk.MLSADigitalFilterStabilityCheck(
        M,
        alpha,
        fft_length=L,
        pade_order=P,
        strict=strict,
        mod_type=mod_type,
        warn_type="ignore",
    )

    opt = "-r 0 " if strict else "-r 1 "
    if mod_type == "fast":
        opt += "-f "

    U.check_compatibility(
        device,
        mlsacheck,
        [],
        f"nrand -l {B*L} | mgcep -m {M} -l {L} -a {alpha} | sopr -m 10",
        f"mlsacheck -m {M} -l {L} -a {alpha} -P {P} {opt} -e 0 -x",
        [],
        dx=M + 1,
        dy=M + 1,
    )

    U.check_differentiable(device, mlsacheck, [B, M + 1])
