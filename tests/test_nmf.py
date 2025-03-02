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


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("beta", [0, 1, 2])
@pytest.mark.parametrize("act_norm", [False, True])
@pytest.mark.parametrize("batch_size", [None, 5])
def test_convergence(
    device, beta, act_norm, batch_size, M=5, T=100, K=3, verbose=False
):
    if device == "cuda" and not torch.cuda.is_available():
        return

    torch.manual_seed(1234)
    x = diffsptk.nrand(T, M, device=device) ** 2
    nmf = diffsptk.NMF(
        T,
        M,
        K,
        beta=beta,
        eps=0.01,
        act_norm=act_norm,
        batch_size=batch_size,
        verbose=verbose,
    ).to(device)
    nmf.warmup(x)
    (U, H), _ = nmf(x)
    y = torch.matmul(U, H)
    error = (x - y).abs().mean()
    assert error < 1

    y = nmf.transform(x)
    assert y.shape == (T, K)
