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

import torch
import pytest

import diffsptk
import tests.utils as U

from scipy.fftpack import dst as scipy_dst


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("module", [False, True])
def test_compatibility(device, module, L=8, B=2):
    if device == "cuda" and not torch.cuda.is_available():
        return

    x = diffsptk.nrand(B, L - 1, device=device)

    dst = U.choice(
        module,
        diffsptk.DST,
        diffsptk.functional.dst,
        {"dst_length": L},
    )
    if hasattr(dst, "to"):
        dst.to(device)

    y1 = scipy_dst(x.cpu().numpy(), norm="ortho")
    y2 = dst(x).cpu().numpy()
    U.allclose(y1, y2)

    U.check_differentiability(device, dst, [B, L])
