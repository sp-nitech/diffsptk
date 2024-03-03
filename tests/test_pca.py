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

import numpy as np
import pytest
import torch

import diffsptk
import tests.utils as U


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("cov_type", [0, 1, 2])
def test_compatibility(device, cov_type, B=10, M=4, N=3):
    if device == "cuda" and not torch.cuda.is_available():
        return

    # C++
    tmp1 = "pca.tmp1"
    tmp2 = "pca.tmp2"
    cmd = (
        f"nrand -l {B*(M+1)} | "
        f"pca -m {M} -n {N} -u {cov_type} -v {tmp1} -d 1e-8 > {tmp2}"
    )
    U.call(cmd, get=False)
    e1 = U.call(f"bcut -e {N-1} {tmp1}")
    v1 = U.call(f"cat {tmp2}").reshape(N + 1, M + 1)
    y1 = U.call(f"nrand -l {B*(M+1)} | pcas -m {M} -n {N} {tmp2}").reshape(-1, N)
    U.call(f"rm {tmp1} {tmp2}", get=False)

    # Python
    pca = diffsptk.PCA(M, N, cov_type=cov_type).to(device)
    x = torch.from_numpy(U.call(f"nrand -l {B*(M+1)}")).reshape(B, M + 1).to(device)
    e, v, m = pca(x)
    e2 = e.cpu().numpy()
    v2 = torch.cat([m.unsqueeze(1), v], dim=1).T.cpu().numpy()
    y2 = pca.transform(x).cpu().numpy()

    assert U.allclose(e1, e2)
    assert U.allclose(np.abs(v1), np.abs(v2))
    assert U.allclose(np.abs(y1), np.abs(y2))
