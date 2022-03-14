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
from tests.utils import call


@pytest.mark.parametrize("n_iter", [0, 3])
def test_compatibility(n_iter, M=8, L=16, B=2, accel=0.001):
    spec = diffsptk.Spectrum(L, eps=0)
    fftcep = diffsptk.CepstralAnalysis(M, L, n_iter=n_iter, accel=accel)
    x = spec(torch.from_numpy(call(f"nrand -l {B*L}").reshape(-1, L)))
    y = fftcep.forward(x).cpu().numpy()

    y_ = call(f"nrand -l {B*L} | fftcep -i {n_iter} -l {L} -m {M} -a {accel}").reshape(
        -1, M + 1
    )
    assert np.allclose(y, y_)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_differentiable(device, n_iter=3, M=4, L=16, B=2):
    if device == "cuda" and not torch.cuda.is_available():
        return

    spec = diffsptk.Spectrum(L)
    fftcep = diffsptk.CepstralAnalysis(M, L, n_iter=n_iter)
    x = torch.randn(B, L, requires_grad=True, device=device)
    y = fftcep.forward(spec(x))

    optimizer = torch.optim.SGD([x], lr=0.001)
    optimizer.zero_grad()
    loss = y.mean()
    loss.backward()
    optimizer.step()
