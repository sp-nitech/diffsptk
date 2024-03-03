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
import tests.utils as U


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("module", [False, True])
@pytest.mark.parametrize("out_format", [0, 1])
def test_compatibility(device, module, out_format, M=12, B=2):
    root_pol = U.choice(
        module,
        diffsptk.PolynomialToRoots,
        diffsptk.functional.root_pol,
        {"order": M},
        {"out_format": out_format},
    )

    def eq(y_hat, y):
        y_hat = np.sort_complex(y_hat)
        y = np.sort_complex(y[0::2] + 1j * y[1::2])
        return U.allclose(y_hat, y)

    U.check_compatibility(
        device,
        root_pol,
        [],
        f"nrand -m {M}",
        f"root_pol -m {M} -i 100 -o {out_format}",
        [],
        eq=eq,
    )

    U.check_differentiability(device, [torch.abs, root_pol], [B, M + 1])
