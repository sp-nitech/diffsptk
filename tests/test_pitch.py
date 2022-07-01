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
@pytest.mark.parametrize("out_format", [0, 1, 2])
def test_compatibility(device, out_format, P=80, sr=16000, L=80, H=240):
    pitch = diffsptk.Pitch(P, sr, f_min=L, f_max=H, out_format=out_format, model="tiny")

    def eq(o):
        if o == 0 or o == 1:
            unvoiced_symbol = 0
            target_error = 3
        elif o == 2:
            unvoiced_symbol = -1e10
            target_error = 0.03
        else:
            raise NotImplementedError

        def inner_eq(y_hat, y):
            idx = np.where(y != unvoiced_symbol)[0]
            error = np.mean(np.abs(y_hat[idx] - y[idx]))
            return error < target_error

        return inner_eq

    U.check_compatibility(
        device,
        [pitch.decode, pitch.forward],
        [],
        "x2x +sd tools/SPTK/asset/data.short",
        f"pitch -s {sr//1000} -p {P} -L {L} -H {H} -o {out_format}",
        [],
        eq=eq(out_format),
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_differentiable(device, P=80, sr=16000, B=2, T=1000):
    pitch = diffsptk.Pitch(P, sr, model="tiny")
    U.check_differentiable(device, pitch, [B, T])
