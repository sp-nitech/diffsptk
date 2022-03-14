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
import torch

import diffsptk
from tests.utils import call


def test_compatibility(T=100, P=10, L1=12, L2=16, eps=1e-6):
    stft = diffsptk.STFT(L1, P, L2, eps=eps)
    x = torch.from_numpy(call(f"nrand -l {T}"))
    y = stft(x).cpu().numpy()

    cmd = (
        f"nrand -l {T} | "
        f"frame -l {L1} -p {P} | "
        f"window -l {L1} -L {L2} | "
        f"spec -l {L2} -e {eps} -o 3"
    )
    y_ = call(cmd).reshape(-1, L2 // 2 + 1)
    assert np.allclose(y, y_)
