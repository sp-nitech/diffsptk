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
@pytest.mark.parametrize("o", [0, 1, 2])
def test_compatibility(device, module, o, B=2, T=100, L=20):
    if o == 0:
        frame_length = None
        reduction = "mean"
        dim = None
        B = 1
    elif o == 1:
        frame_length = L
        reduction = "mean"
        dim = None
    elif o == 2:
        frame_length = L
        reduction = "none"
        dim = T // L

    snr = U.choice(
        module,
        diffsptk.SNR,
        diffsptk.functional.snr,
        {},
        {"frame_length": frame_length, "full": True, "reduction": reduction},
        n_input=2,
    )

    tmp1 = "snr.tmp1"
    tmp2 = "snr.tmp2"
    U.check_compatibility(
        device,
        snr,
        [f"nrand -s 1 -l {B*T} > {tmp1}", f"nrand -s 2 -l {B*T} > {tmp2}"],
        [f"cat {tmp1}", f"cat {tmp2}"],
        f"snr -o {o} -l {L} {tmp1} {tmp2}",
        [f"rm {tmp1} {tmp2}"],
        dx=T,
        dy=dim,
    )

    U.check_differentiability(device, snr, [(B, T), (B, T)])
