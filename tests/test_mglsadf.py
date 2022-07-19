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

import os

import numpy as np
import pytest

import diffsptk
import tests.utils as U


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("ignore_gain", [False, True])
@pytest.mark.parametrize("c", [0, 10])
def test_compatibility(device, ignore_gain, c, alpha=0.42, M=24, P=80):
    mglsadf = diffsptk.MLSA(
        M,
        cep_order=100,
        taylor_order=20,
        frame_period=P,
        alpha=alpha,
        c=c,
        ignore_gain=ignore_gain,
    )

    tmp1 = "mglsadf.tmp1"
    tmp2 = "mglsadf.tmp2"
    opt = "-k" if ignore_gain else ""
    cmd = (
        f"x2x +sd tools/SPTK/asset/data.short | "
        f"frame -p {P} -l 400 | "
        f"window -w 1 -n 1 -l 400 -L 512 | "
        f"mgcep -c {c} -a {alpha} -m {M} -l 512 -E -60 > {tmp2}"
    )
    T = os.path.getsize("tools/SPTK/asset/data.short") // 2
    U.check_compatibility(
        device,
        mglsadf,
        [f"nrand -l {T} > {tmp1}", cmd],
        [f"cat {tmp1}", f"cat {tmp2}"],
        f"mglsadf {tmp2} < {tmp1} -i 1 -m {M} -p {P} -c {c} -a {alpha} {opt}",
        [f"rm {tmp1} {tmp2}"],
        dx=[None, M + 1],
        eq=lambda a, b: np.corrcoef(a, b)[0, 1] > 0.99,
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_differentiable(device, B=4, T=20, M=4):
    mglsadf = diffsptk.MLSA(M, taylor_order=5)
    U.check_differentiable(device, mglsadf, [(B, T), (B, T, M + 1)])
