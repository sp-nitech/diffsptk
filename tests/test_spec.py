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


@pytest.mark.parametrize("out_format", [0, 1, 2, 3])
def test_compatibility(out_format, L=16, B=2, eps=0.01):
    spec = diffsptk.Spectrum(L, out_format=out_format, eps=eps)
    x = torch.from_numpy(call(f"nrand -l {B*L}").reshape(-1, L))
    y = spec(x).cpu().numpy()

    y_ = call(f"nrand -l {B*L} | spec -l {L} -o {out_format} -e {eps}").reshape(
        -1, L // 2 + 1
    )
    assert np.allclose(y, y_)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_differentiable(device, L=16, B=2):
    if device == "cuda" and not torch.cuda.is_available():
        return

    spec = diffsptk.Spectrum(L).to(device)
    x = torch.randn(B, L, requires_grad=True, device=device)
    check(spec, x)
