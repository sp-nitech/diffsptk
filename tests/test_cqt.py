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


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_compatibility(device, f_min=100, sr=16000, T=1000):
    if device == "cuda" and not torch.cuda.is_available():
        return

    x = diffsptk.sin(T - 1, period=sr / f_min).to(device)
    cqt = diffsptk.ConstantQTransform(T, sr, f_min=f_min).to(device)
    X = cqt(x)
    assert torch.argmax(X.abs()) == 0

    U.check_differentiable(device, [torch.abs, cqt], [T])
