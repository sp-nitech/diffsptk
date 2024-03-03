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
@pytest.mark.parametrize("mode", ["multi-stage", "single-stage", "freq-domain"])
@pytest.mark.parametrize("c", [0, 10])
def test_compatibility(
    device, ignore_gain, mode, c, alpha=0.42, M=24, P=80, L=400, fft_length=512
):
    if mode == "multi-stage":
        params = {"cep_order": 100}
    elif mode == "single-stage":
        params = {"ir_length": 200, "n_fft": 512}
    elif mode == "freq-domain":
        params = {"frame_length": L, "fft_length": fft_length}

    mglsadf = diffsptk.MLSA(
        M,
        P,
        alpha=alpha,
        c=c,
        ignore_gain=ignore_gain,
        phase="minimum",
        mode=mode,
        **params,
    )

    tmp1 = "mglsadf.tmp1"
    tmp2 = "mglsadf.tmp2"
    T = os.path.getsize("tools/SPTK/asset/data.short") // 2
    cmd1 = f"nrand -l {T} > {tmp1}"
    cmd2 = (
        f"x2x +sd tools/SPTK/asset/data.short | "
        f"frame -p {P} -l {L} | "
        f"window -w 1 -n 1 -l {L} -L {fft_length} | "
        f"mgcep -c {c} -a {alpha} -m {M} -l {fft_length} -E -60 > {tmp2}"
    )
    opt = "-k" if ignore_gain else ""
    U.check_compatibility(
        device,
        mglsadf,
        [cmd1, cmd2],
        [f"cat {tmp1}", f"cat {tmp2}"],
        f"mglsadf {tmp2} < {tmp1} -m {M} -p {P} -c {c} -a {alpha} {opt}",
        [f"rm {tmp1} {tmp2}"],
        dx=[None, M + 1],
        eq=lambda a, b: np.corrcoef(a, b)[0, 1] > 0.98,
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("ignore_gain", [False, True])
@pytest.mark.parametrize("phase", ["minimum", "maximum", "zero"])
@pytest.mark.parametrize("mode", ["multi-stage", "single-stage", "freq-domain"])
def test_differentiable(device, ignore_gain, phase, mode, B=4, T=20, P=2, M=4):
    if mode == "multi-stage":
        params = {"cep_order": 10}
    elif mode == "single-stage":
        params = {"ir_length": 20, "n_fft": 32}
    elif mode == "freq-domain":
        params = {"frame_length": 6, "fft_length": 16}

    mglsadf = diffsptk.MLSA(
        M, P, ignore_gain=ignore_gain, phase=phase, mode=mode, **params
    )
    U.check_differentiability(device, mglsadf, [(B, T), (B, T // P, M + 1)])
