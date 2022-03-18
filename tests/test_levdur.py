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
from tests.utils import compose


def test_compatibility(M=30, L=52, B=1):
    acorr = diffsptk.AutocorrelationAnalysis(M, L)
    levdur = diffsptk.LevinsonDurbinRecursion()
    x = acorr(torch.from_numpy(call(f"nrand -l {B*L}").reshape(-1, L)))
    y = levdur.forward(x, n_out=1).cpu().numpy()

    y_ = call(f"nrand -l {B*L} | lpc -m {M} -l {L}").reshape(-1, M + 1)
    assert np.allclose(y, y_)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_differentiable(device, M=30, L=52, B=2):
    if device == "cuda" and not torch.cuda.is_available():
        return

    acorr = diffsptk.AutocorrelationAnalysis(M, L).to(device)
    levdur = diffsptk.LevinsonDurbinRecursion().to(device)
    x = torch.randn(B, L, requires_grad=True, device=device)
    check(compose(levdur, acorr), x, opt={"n_out": 1})
