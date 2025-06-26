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

import diffsptk
import tests.utils as U


@pytest.mark.parametrize("module", [False, True])
@pytest.mark.parametrize("wht_type", [1, 2, 3])
def test_compatibility(device, dtype, module, wht_type, L=8, B=2):
    wht = U.choice(
        module,
        diffsptk.WHT,
        diffsptk.functional.wht,
        {"wht_length": L, "wht_type": wht_type, "device": device, "dtype": dtype},
    )

    if module:
        if wht_type == 1:
            H = np.array(
                [
                    [1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, -1, -1, -1, -1],
                    [1, 1, -1, -1, -1, -1, 1, 1],
                    [1, 1, -1, -1, 1, 1, -1, -1],
                    [1, -1, -1, 1, 1, -1, -1, 1],
                    [1, -1, -1, 1, -1, 1, 1, -1],
                    [1, -1, 1, -1, -1, 1, -1, 1],
                    [1, -1, 1, -1, 1, -1, 1, -1],
                ]
            )
        elif wht_type == 2:
            H = np.array(
                [
                    [1, 1, 1, 1, 1, 1, 1, 1],
                    [1, -1, 1, -1, 1, -1, 1, -1],
                    [1, 1, -1, -1, 1, 1, -1, -1],
                    [1, -1, -1, 1, 1, -1, -1, 1],
                    [1, 1, 1, 1, -1, -1, -1, -1],
                    [1, -1, 1, -1, -1, 1, -1, 1],
                    [1, 1, -1, -1, -1, -1, 1, 1],
                    [1, -1, -1, 1, -1, 1, 1, -1],
                ]
            )
        elif wht_type == 3:
            H = np.array(
                [
                    [1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, -1, -1, -1, -1],
                    [1, 1, -1, -1, 1, 1, -1, -1],
                    [1, 1, -1, -1, -1, -1, 1, 1],
                    [1, -1, 1, -1, 1, -1, 1, -1],
                    [1, -1, 1, -1, -1, 1, -1, 1],
                    [1, -1, -1, 1, 1, -1, -1, 1],
                    [1, -1, -1, 1, -1, 1, 1, -1],
                ]
            )
        W = np.round(wht.W.cpu().numpy() * 2 ** (np.log2(L) / 2))
        assert U.allclose(W, H)

    U.check_differentiability(device, dtype, wht, [B, L])
