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
@pytest.mark.parametrize("unwrap", [False, True])
def test_compatibility(device, module, unwrap, L=16, B=2):
    phase = U.choice(
        module,
        diffsptk.Phase,
        diffsptk.functional.phase,
        {},
        {"fft_length": L, "unwrap": unwrap},
        n_input=2,
    )

    opt = "-u" if unwrap else ""
    tmp1 = "phase.tmp1"
    tmp2 = "phase.tmp2"
    U.check_compatibility(
        device,
        phase,
        [f"nrand -s 1 -l {B*L} > {tmp1}", f"nrand -s 2 -l {B*L} > {tmp2}"],
        [f"cat {tmp1}", f"cat {tmp2}"],
        f"phase -l {L} -m {L-1} -z {tmp1} -n {L-1} -p {tmp2} {opt}",
        [f"rm {tmp1} {tmp2}"],
        dx=L,
        dy=L // 2 + 1,
    )

    U.check_differentiability(device, phase, [(B, L), (B, L)])


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_compatibility_b(device, L=16, B=2):
    phase = diffsptk.Phase(L)

    U.check_compatibility(
        device,
        phase,
        [],
        f"nrand -s 1 -l {B*L}",
        f"phase -l {L} -m {L-1}",
        [],
        dx=L,
        dy=L // 2 + 1,
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_compatibility_a(device, L=16, B=2):
    phase = diffsptk.Phase(L)

    tmp = "phase.tmp"
    U.check_compatibility(
        device,
        lambda x: phase(b=None, a=x),
        [f"nrand -s 2 -l {B*L} > {tmp}"],
        [f"cat {tmp}"],
        f"phase -l {L} -n {L-1} -p {tmp}",
        [f"rm {tmp}"],
        dx=L,
        dy=L // 2 + 1,
    )
