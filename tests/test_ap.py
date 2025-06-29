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


@pytest.mark.skip_float_check
@pytest.mark.parametrize("algorithm", ["tandem", "d4c"])
@pytest.mark.parametrize("out_format", [0, 1, 2, 3])
def test_compatibility(
    device, dtype, algorithm, out_format, M=24, P=80, sr=16000, L=512, B=2
):
    if algorithm == "tandem":
        params = {}
    elif algorithm == "d4c":
        params = {"threshold": 0.01}

    ap = diffsptk.Aperiodicity(
        P, sr, L, algorithm, out_format=out_format, device=device, dtype=dtype, **params
    )

    def eq(a, o):
        if a == 1:
            return None

        def convert(x, o):
            if o == 0 or o == 1:
                return x
            elif o == 2 or o == 3:
                return x / (1 + x)
            else:
                raise ValueError

        target_ap_error = 0.2

        def inner_eq(y_hat, y):
            y_hat = convert(y_hat, o)
            y = convert(y, o)
            return np.mean(np.abs(y_hat - y)) < target_ap_error

        return inner_eq

    s = sr // 1000
    a = 0 if algorithm == "tandem" else 1
    tmp1 = "ap.tmp1"
    tmp2 = "ap.tmp2"
    U.check_compatibility(
        device,
        dtype,
        ap,
        [
            f"x2x +sd tools/SPTK/asset/data.short > {tmp1}",
            f"pitch -s {s} -p {P} -L 80 -H 180 -o 1 {tmp1} > {tmp2}",
        ],
        [f"cat {tmp1}", f"cat {tmp2}"],
        f"ap -s {s} -p {P} -l {L} -a {a} -q 1 -o {out_format} {tmp2} < {tmp1}",
        [f"rm {tmp1} {tmp2}"],
        dy=L // 2 + 1,
        eq=eq(a, out_format),
    )

    U.check_differentiability(
        device, dtype, ap, [(B, sr), (B, sr // P)], checks=[True, False]
    )
