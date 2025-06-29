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

from operator import itemgetter

import diffsptk
import tests.utils as U


def test_compatibility(device, dtype, m=9, K=4, B=8):
    vq = diffsptk.VectorQuantization(m, K, device=device)

    tmp1 = "vq.tmp1"
    tmp2 = "vq.tmp2"
    U.check_compatibility(
        device,
        dtype,
        [itemgetter(1), vq],
        [
            f"nrand -s 123 -l {B * (m + 1)} > {tmp1}",
            f"nrand -s 234 -l {K * (m + 1)} > {tmp2}",
        ],
        [f"cat {tmp1}", f"cat {tmp2}"],
        f"msvq -m {m} -s {tmp2} < {tmp1} | x2x +id",
        [f"rm {tmp1} {tmp2}"],
        dx=[m + 1, m + 1],
    )

    U.check_differentiability(device, dtype, [itemgetter(2), vq], [m + 1])
