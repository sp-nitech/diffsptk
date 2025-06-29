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


@pytest.mark.parametrize("cov_type", [0, 1, 2])
@pytest.mark.parametrize("batch_size", [None, 5])
def test_compatibility(device, dtype, cov_type, batch_size, B=10, M=4, K=3):
    # C++
    tmp1 = "pca.tmp1"
    tmp2 = "pca.tmp2"
    cmd = (
        f"nrand -l {B * (M + 1)} | "
        f"pca -m {M} -n {K} -u {cov_type} -v {tmp1} -d 1e-8 > {tmp2}"
    )
    U.call(cmd, get=False)
    s1 = U.call(f"bcut -e {K - 1} {tmp1}")
    v1 = U.call(f"cat {tmp2}").reshape(K + 1, M + 1)
    y1 = U.call(f"nrand -l {B * (M + 1)} | pcas -m {M} -n {K} {tmp2}").reshape(-1, K)
    U.call(f"rm {tmp1} {tmp2}", get=False)

    # Python
    pca = diffsptk.PCA(
        M, K, cov_type=cov_type, batch_size=batch_size, device=device, dtype=dtype
    )
    x = (
        torch.from_numpy(U.call(f"nrand -l {B * (M + 1)}"))
        .reshape(B, M + 1)
        .to(device=device, dtype=torch.get_default_dtype() if dtype is None else dtype)
    )
    s, v, m = pca(x)
    s2 = s.cpu().numpy()
    v2 = torch.cat([m.unsqueeze(0), v], dim=0).cpu().numpy()
    y2 = pca.transform(x).cpu().numpy()

    assert U.allclose(s1, s2)
    assert U.allclose(np.abs(v1), np.abs(v2))
    assert U.allclose(np.abs(y1), np.abs(y2))

    z = pca.center(x)
    assert U.allclose(torch.mean(z, dim=0).cpu().numpy(), np.zeros(M + 1))
    if cov_type <= 1:
        z = pca.whiten(x)
        assert U.allclose(torch.cov(z.T, correction=cov_type).cpu().numpy(), np.eye(K))
