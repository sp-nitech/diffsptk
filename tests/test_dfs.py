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
@pytest.mark.parametrize("mode", ["fir", "iir"])
def test_compatibility(device, mode, b=[-0.42, 1], a=[1, -0.42], T=100):
    dfs = diffsptk.IIR(b, a, ir_length=30, mode=mode)

    bb = " ".join([str(x) for x in b])
    aa = " ".join([str(x) for x in a])

    U.check_compatibility(
        device,
        dfs,
        [],
        f"nrand -l {T}",
        f"dfs -b {bb} -a {aa}",
        [],
    )

    U.check_differentiability(device, dfs, [T])


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("mode", ["fir", "iir"])
def test_compatibility_b(device, mode, b=[-0.42, 1], T=100):
    dfs = diffsptk.IIR(b, None, mode=mode)

    bb = " ".join([str(x) for x in b])

    tmp1 = "dfs.tmp1"
    tmp2 = "dfs.tmp2"
    U.check_compatibility(
        device,
        dfs,
        [f"nrand -l {T} > {tmp1}", f"echo {bb} | x2x +ad > {tmp2}"],
        [f"cat {tmp1}", f"cat {tmp2}"],
        f"dfs -z {tmp2} < {tmp1}",
        [f"rm {tmp1} {tmp2}"],
    )

    U.check_differentiability(device, dfs, [(T,), (len(b),)])


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("mode", ["iir"])
def test_compatibility_b_a(device, mode, b=[-0.42, 1], a=[1, -0.42, 0], T=100):
    dfs = diffsptk.IIR(None, None, mode=mode)

    bb = " ".join([str(x) for x in b])
    aa = " ".join([str(x) for x in a])

    tmp1 = "dfs.tmp1"
    tmp2 = "dfs.tmp2"
    tmp3 = "dfs.tmp3"
    U.check_compatibility(
        device,
        dfs,
        [
            f"nrand -l {T} > {tmp1}",
            f"echo {bb} | x2x +ad > {tmp2}",
            f"echo {aa} | x2x +ad > {tmp3}",
        ],
        [f"cat {tmp1}", f"cat {tmp2}", f"cat {tmp3}"],
        f"dfs -z {tmp2} -p {tmp3} < {tmp1}",
        [f"rm {tmp1} {tmp2} {tmp3}"],
    )

    U.check_differentiability(device, dfs, [(T,), (len(b),), (len(a),)])


def test_various_shape(T=10):
    pqmf = diffsptk.IIR()
    U.check_various_shape(pqmf, [(T,), (1, T), (1, 1, T)])
