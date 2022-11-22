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
@pytest.mark.parametrize("cascade", [False, True])
def test_compatibility(device, ignore_gain, cascade, c=0, alpha=0.42, M=24, P=80):
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

    if cascade:
        params = {"cep_order": 100, "taylor_order": 40}
    else:
        params = {"impulse_response_length": 500, "n_fft": 1024}

    # Get residual signal.
    imglsadf = diffsptk.IMLSA(
        M,
        frame_period=P,
        alpha=alpha,
        c=c,
        ignore_gain=ignore_gain,
        cascade=cascade,
        phase="minimum",
        **params,
    ).to(device)
    x = imglsadf(y, mc)

    # Get reconstructed signal.
    y_hat = imglsadf(x, -mc)

    # Compute correlation between two signals.
    r = torch.corrcoef(torch.stack([y, y_hat]))[0, 1]
    assert torch.gt(r, 0.99)
