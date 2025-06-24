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
@pytest.mark.parametrize("out_format", [0, 1, 2, 3, 4])
def test_compatibility(device, dtype, module, out_format, M=12, L=16, B=2):
    fftr = U.choice(
        module,
        diffsptk.RealValuedFastFourierTransform,
        diffsptk.functional.fftr,
        {
            "fft_length": L,
            "out_format": out_format,
            "learnable": "debug",
            "device": device,
            "dtype": dtype,
        },
    )

    size = L // 2 + 1

    def eq(o):
        def inner_eq(y_hat, y):
            if o == 0:
                y = y[..., :size] + 1j * y[..., size:]
            return U.allclose(y_hat, y)

        return inner_eq

    U.check_compatibility(
        device,
        dtype,
        fftr,
        [],
        f"nrand -l {B * (M + 1)}",
        f"fftr -m {M} -l {L} -o {out_format} -H",
        [],
        dx=M + 1,
        dy=2 * size if out_format == 0 else size,
        eq=eq(out_format),
    )

    U.check_differentiability(device, dtype, [torch.abs, fftr], [B, L])


def test_learnable(L=16):
    fftr = diffsptk.RealValuedFastFourierTransform(
        L, learnable=True, out_format="amplitude"
    )
    U.check_learnable(fftr, (L,))
