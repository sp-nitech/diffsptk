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
def test_compatibility(device, dtype, module, L=16, alpha=0.4, gamma=0.9, B=2):
    grpdelay = U.choice(
        module,
        diffsptk.GroupDelay,
        diffsptk.functional.grpdelay,
        {
            "fft_length": L,
            "alpha": alpha,
            "gamma": gamma,
            "device": device,
            "dtype": dtype,
        },
    )

    tmp1 = "grpdelay.tmp1"
    tmp2 = "grpdelay.tmp2"
    M = L // 2
    N = L // 4
    U.check_compatibility(
        device,
        dtype,
        grpdelay,
        [f"nrand -s 1 -l {B * M} > {tmp1}", f"nrand -s 2 -l {B * N} > {tmp2}"],
        [f"cat {tmp1}", f"cat {tmp2}"],
        (
            f"grpdelay -l {L} -m {M - 1} -z {tmp1} -n {N - 1} -p {tmp2} "
            f"-a {alpha} -g {gamma}"
        ),
        [f"rm {tmp1} {tmp2}"],
        dx=[M, N],
        dy=L // 2 + 1,
    )

    U.check_differentiability(device, dtype, grpdelay, [(B, M), (B, N)])


def test_compatibility_b(device, dtype, L=16, B=2):
    grpdelay = diffsptk.GroupDelay(L, device=device, dtype=dtype)

    M = L // 2
    U.check_compatibility(
        device,
        dtype,
        grpdelay,
        [],
        f"nrand -s 1 -l {B * M}",
        f"grpdelay -l {L} -m {M - 1}",
        [],
        dx=M,
        dy=L // 2 + 1,
    )


def test_compatibility_a(device, dtype, L=16, B=2):
    grpdelay = diffsptk.GroupDelay(L, device=device, dtype=dtype)

    tmp = "grpdelay.tmp"
    N = L // 4
    U.check_compatibility(
        device,
        dtype,
        grpdelay,
        [f"nrand -s 2 -l {B * N} > {tmp}"],
        [f"cat {tmp}"],
        f"grpdelay -l {L} -n {N - 1} -p {tmp}",
        [f"rm {tmp}"],
        dx=N,
        dy=L // 2 + 1,
        key=["a"],
    )
