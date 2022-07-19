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


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("out_format", [0, 1, 2, 3])
@pytest.mark.parametrize("relative_floor", [None, -40])
def test_compatibility(device, out_format, relative_floor, L=16, B=2, eps=0.01):
    spec = diffsptk.Spectrum(
        L, out_format=out_format, eps=eps, relative_floor=relative_floor
    )

    opt = f"-E {relative_floor}" if relative_floor is not None else ""
    U.check_compatibility(
        device,
        spec,
        [],
        f"nrand -l {B*L}",
        f"spec -l {L} -o {out_format} -e {eps} {opt}",
        [],
        dx=L,
        dy=L // 2 + 1,
    )

    tmp1 = "spec.tmp1"
    tmp2 = "spec.tmp2"
    U.check_compatibility(
        device,
        spec,
        [f"nrand -s 1 -l {B*L} > {tmp1}", f"nrand -s 2 -l {B*L} > {tmp2}"],
        [f"cat {tmp1}", f"cat {tmp2}"],
        (
            f"spec -l {L} -o {out_format} -e {eps} {opt} "
            f"-m {L-1} -z {tmp1} -n {L-1} -p {tmp2}"
        ),
        [f"rm {tmp1} {tmp2}"],
        dx=L,
        dy=L // 2 + 1,
    )

    U.check_differentiable(device, spec, [(B, L), (B, L)])
