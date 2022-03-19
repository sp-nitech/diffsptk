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


def test_compatibility(M=14, L=30, B=2):
    lpc = diffsptk.LPC(M, L, out_format="Ka")
    x = torch.from_numpy(call(f"nrand -l {B*L}")).reshape(-1, L)
    y = lpc(x).cpu().numpy()

    y_ = call(f"nrand -l {B*L} | lpc -l {L} -m {M}").reshape(-1, M + 1)
    assert np.allclose(y, y_)
