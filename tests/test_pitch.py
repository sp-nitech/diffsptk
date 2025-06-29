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


@pytest.mark.parametrize("algorithm", ["crepe", "fcnf0"])
def test_probability_calculation(algorithm, P=80, L=80, H=180):
    x, sr = diffsptk.read("assets/data.wav")

    try:
        pitch = diffsptk.Pitch(P, sr, algorithm, f_min=L, f_max=H, out_format="prob")
    except ImportError:
        pytest.skip(f"Algorithm '{algorithm}' is not available.")

    prob = pitch(x)
    assert prob.dim() == 2


@pytest.mark.parametrize("algorithm", ["crepe", "fcnf0"])
@pytest.mark.parametrize("out_format", [0, 1, 2])
def test_compatibility(
    device, dtype, algorithm, out_format, P=80, sr=16000, L=80, H=180
):
    try:
        pitch = diffsptk.Pitch(
            P,
            sr,
            algorithm,
            f_min=L,
            f_max=H,
            out_format=out_format,
            device=device,
            dtype=dtype,
        )
    except ImportError:
        pytest.skip(f"Algorithm '{algorithm}' is not available.")

    def eq(o, sr):
        def convert(x, o):
            if o == 0:
                x[x != 0] = sr / x[x != 0]
            elif o == 1:
                pass
            elif o == 2:
                x[x != -1e10] = np.exp(x[x != -1e10])
                x[x == -1e10] = 0
            else:
                raise ValueError
            return x

        unvoiced_symbol = 0
        target_f0_error = 5
        target_uv_error = 35

        def inner_eq(y_hat, y):
            y_hat = convert(y_hat, o)
            y = convert(y, o)

            idx = np.where(
                np.logical_and(y_hat != unvoiced_symbol, y != unvoiced_symbol)
            )[0]
            f0_error = np.mean(np.abs(y_hat[idx] - y[idx]))

            idx1 = np.logical_and(y_hat != unvoiced_symbol, y == unvoiced_symbol)
            idx2 = np.logical_and(y_hat == unvoiced_symbol, y != unvoiced_symbol)
            uv_error = np.sum(idx1) + np.sum(idx2)

            return f0_error < target_f0_error and uv_error < target_uv_error

        return inner_eq

    U.check_compatibility(
        device,
        dtype,
        [pitch, lambda x: x / 32768],
        [],
        "x2x +sd tools/SPTK/asset/data.short",
        f"pitch -s {sr // 1000} -p {P} -L {L} -H {H} -o {out_format}",
        [],
        eq=eq(out_format, sr),
    )
