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

from operator import itemgetter

import pytest
import torch

import diffsptk
import tests.utils as U


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("batch_size", [None, 5])
def test_compatibility(device, batch_size, M=1, K=4, B=10, n_iter=10):
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)
    lbg = diffsptk.LBG(M, K, n_iter=n_iter, batch_size=batch_size)

    tmp1 = "lbg.tmp1"
    tmp2 = "lbg.tmp2"
    tmp3 = "lbg.tmp3"
    tmp4 = "lbg.tmp4"
    tmp5 = "lbg.tmp5"
    U.check_compatibility(
        device,
        [itemgetter(-1), lbg],
        [
            f"nrand -u +2 -l {B*(M+1)} -s 1 > {tmp1}",
            f"nrand -u -2 -l {B*(M+1)} -s 2 > {tmp2}",
            f"nrand -u +4 -l {B*(M+1)} -s 3 > {tmp3}",
            f"nrand -u -4 -l {B*(M+1)} -s 4 > {tmp4}",
        ],
        f"cat {tmp1} {tmp2} {tmp3} {tmp4}",
        (
            f"cat {tmp1} {tmp2} {tmp3} {tmp4} | "
            f"lbg -m {M} -e {K} -i {n_iter} -s 1234 -S {tmp5} > /dev/null; "
            f"cat {tmp5}"
        ),
        [f"rm {tmp1} {tmp2} {tmp3} {tmp4} {tmp5}"],
        dx=M + 1,
        rtol=0.1,
    )


def test_special_case(M=1, K=4, B=10, n_iter=10):
    torch.manual_seed(1234)
    x = torch.randn(B, M + 1)
    lbg = diffsptk.LBG(
        M, K, n_iter=n_iter, min_data_per_cluster=int(B * 0.9), init="none"
    )
    _, idx1, dist = lbg(x, return_indices=True)
    _, idx2 = lbg.transform(x)
    assert torch.all(idx1 == idx2)

    extra_lbg = diffsptk.LBG(M, K * 2, n_iter=n_iter, init=lbg.vq.codebook)
    _, extra_dist = extra_lbg(x)
    assert extra_dist < dist
