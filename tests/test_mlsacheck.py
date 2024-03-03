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
@pytest.mark.parametrize("P", [4, 5, 6, 7])
@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize("threshold", [None, 4])
@pytest.mark.parametrize("fast", [False, True])
@pytest.mark.parametrize("mod_type", ["clip", "scale"])
def test_compatibility(
    device, module, P, strict, threshold, fast, mod_type, M=9, L=32, alpha=0.1, B=10
):
    if fast and mod_type == "clip":
        return
    if threshold is not None and P != 4 and strict:
        return

    mlsacheck = U.choice(
        module,
        diffsptk.MLSADigitalFilterStabilityCheck,
        diffsptk.functional.mlsacheck,
        {"cep_order": M},
        {
            "alpha": alpha,
            "pade_order": P,
            "strict": strict,
            "threshold": threshold,
            "fast": fast,
            "n_fft": L,
            "warn_type": "ignore",
            "mod_type": mod_type,
        },
    )

    opt = "-f " if fast else ""
    opt += "-r 0 " if strict else "-r 1 "
    opt += "-t 0 " if mod_type == "clip" else "-t 1 "
    opt += f"-R {threshold} " if threshold is not None else ""
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

    U.check_differentiability(device, mlsacheck, [B, M + 1])
