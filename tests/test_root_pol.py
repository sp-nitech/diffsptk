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


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("out_format", [0, 1])
def test_compatibility(device, out_format, M=12, B=2, n_iter=100):
    root_pol = diffsptk.DurandKernerMethod(M, n_iter=n_iter, out_format=out_format)

    def eq(y_hat, y):
        re = np.real(y_hat).flatten()
        im = np.imag(y_hat).flatten()
        y2 = np.empty((re.size + im.size,), dtype=y.dtype)
        y2[0::2] = re
        y2[1::2] = im
        return U.allclose(y2, y)

    U.check_compatibility(
        device,
        [lambda x: x[0], root_pol],
        [],
        f"nrand -m {M}",
        f"root_pol -m {M} -i {n_iter} -o {out_format}",
        [],
        eq=eq,
    )

    U.check_differentiable(device, [lambda x: x[0], root_pol], [B, M + 1])
