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
@pytest.mark.parametrize("cond", [0, 1, 2, 3])
def test_compatibility(device, dtype, module, cond, B=2, T=100, L=20):
    if cond == 0:
        # SNR
        o, reduction, frame_length, dim, mul, B = (0, "mean", None, None, 1, 1)
    elif cond == 1:
        # Segmental SNR
        o, reduction, frame_length, dim, mul = (1, "mean", L, None, 1)
    elif cond == 2:
        # Segmental SNR
        o, reduction, frame_length, dim, mul = (1, "sum", L, None, T // L * B)
    elif cond == 3:
        # Segmental SNR per frame
        o, reduction, frame_length, dim, mul = (2, "none", L, T // L, 1)
    else:
        raise ValueError

    snr = U.choice(
        module,
        diffsptk.SNR,
        diffsptk.functional.snr,
        {"frame_length": frame_length, "full": True, "reduction": reduction},
    )

    tmp1 = "snr.tmp1"
    tmp2 = "snr.tmp2"
    U.check_compatibility(
        device,
        dtype,
        snr,
        [f"nrand -s 1 -l {B * T} > {tmp1}", f"nrand -s 2 -l {B * T} > {tmp2}"],
        [f"cat {tmp1}", f"cat {tmp2}"],
        f"snr -o {o} -l {L} {tmp1} {tmp2} | sopr -m {mul}",
        [f"rm {tmp1} {tmp2}"],
        dx=T,
        dy=dim,
    )

    U.check_differentiability(device, dtype, snr, [(B, T), (B, T)])
