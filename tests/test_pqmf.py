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
from tests.utils import check


@pytest.mark.parametrize("m", [10, 11])
def test_compatibility(m, n_band=4, L=20):
    pqmf = diffsptk.PQMF(n_band, filter_order=m).double()
    x = torch.from_numpy(call(f"nrand -l {L}", double=True).reshape(-1, L))
    y = pqmf(x).cpu().numpy()

    y_ = call(f"nrand -l {L} | pqmf -k {n_band} -m {m}").reshape(-1, n_band).T
    assert np.allclose(y, y_)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_differentiable(device, n_band=4, m=8, B=2, L=20):
    if device == "cuda" and not torch.cuda.is_available():
        return

    pqmf = diffsptk.PQMF(n_band, filter_order=m)
    x = torch.randn(B, L, requires_grad=True, device=device)
    check(pqmf, x)
