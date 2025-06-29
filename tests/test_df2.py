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

import diffsptk
import tests.utils as U


@pytest.mark.parametrize("ir_length", [None, 100])
def test_compatibility(
    device, dtype, ir_length, sr=16000, pf=2000, pb=200, zf=1000, zb=100, T=100
):
    df2 = diffsptk.SecondOrderDigitalFilter(
        **{
            "sample_rate": sr,
            "pole_frequency": pf,
            "pole_bandwidth": pb,
            "zero_frequency": zf,
            "zero_bandwidth": zb,
            "ir_length": ir_length,
            "device": device,
            "dtype": dtype,
        },
    )

    U.check_compatibility(
        device,
        dtype,
        df2,
        [],
        f"nrand -l {T}",
        f"df2 -s {sr // 1000} -p {pf} {pb} -z {zf} {zb}",
        [],
    )

    U.check_differentiability(device, dtype, df2, [T])
