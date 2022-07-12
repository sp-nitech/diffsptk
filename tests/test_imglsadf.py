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
import torch

import diffsptk
import tests.utils as U


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("ignore_gain", [False, True])
def test_compatibility(device, ignore_gain, c=0, alpha=0.42, M=24, P=80):
    if device == "cuda" and not torch.cuda.is_available():
        return

    # Prepare data for C++.
    tmp1 = "mglsadf.tmp1"
    tmp2 = "mglsadf.tmp2"
    U.call(f"x2x +sd tools/SPTK/asset/data.short > {tmp1}", get=False)
    cmd = (
        f"x2x +sd tools/SPTK/asset/data.short | "
        f"frame -p {P} -l 400 | "
        f"window -w 1 -n 1 -l 400 -L 512 | "
        f"mgcep -c {c} -a {alpha} -m {M} -l 512 > {tmp2}"
    )
    U.call(f"{cmd}", get=False)

    # Prepare data for C++.
    y = torch.from_numpy(U.call(f"cat {tmp1}")).to(device)
    mc = torch.from_numpy(U.call(f"cat {tmp2}").reshape(-1, M + 1)).to(device)
    U.call(f"rm {tmp1} {tmp2}", get=False)

    # Get residual signal.
    imglsadf = diffsptk.IMLSA(
        M,
        cep_order=100,
        taylor_order=50,
        frame_period=P,
        alpha=alpha,
        c=c,
        ignore_gain=ignore_gain,
    ).to(device)
    x = imglsadf(y, mc)

    # Get reconstructed signal.
    mglsadf = diffsptk.MLSA(
        M,
        cep_order=100,
        taylor_order=30,
        frame_period=P,
        alpha=alpha,
        c=c,
        ignore_gain=ignore_gain,
    ).to(device)
    y_hat = mglsadf(x, mc)

    # Compute error between two signals.
    error = torch.max(torch.abs(y - y_hat))
    assert torch.lt(error, 1)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_differentiable(device, B=4, T=20, M=4):
    imglsadf = diffsptk.IMLSA(M, taylor_order=5)
    U.check_differentiable(device, imglsadf, [(B, T), (B, T, M + 1)])
