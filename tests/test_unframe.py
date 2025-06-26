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


@pytest.mark.parametrize("module", [False, True])
@pytest.mark.parametrize("fl", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("fp", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("center", [False, True])
def test_compatibility(device, dtype, module, fl, fp, center, T=20):
    if fl < fp:
        return

    frame = diffsptk.Frame(
        fl,
        fp,
        center=center,
    )
    unframe = U.choice(
        module,
        diffsptk.Unframe,
        diffsptk.functional.unframe,
        {
            "frame_length": fl,
            "frame_period": fp,
            "center": center,
            "device": device,
            "dtype": dtype,
        },
    )

    x1 = diffsptk.ramp(T, device=device, dtype=dtype)
    y = frame(x1)
    x2 = diffsptk.ramp(torch.max(y), device=device, dtype=dtype)
    x3 = unframe(y, out_length=x2.size(0))
    assert torch.allclose(x2, x3)

    U.check_differentiability(device, dtype, unframe, [T // fp, fl])


def test_multi_dimensional(fl=5, fp=3, B=2, C=4, T=20):
    x = torch.randn(B, C, T)
    y = diffsptk.functional.frame(x, fl, fp)
    z = diffsptk.functional.unframe(y, out_length=T, frame_period=fp)
    assert torch.allclose(x, z)
