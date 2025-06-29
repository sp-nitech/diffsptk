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

import diffsptk
import tests.utils as U


def test_compatibility(device, dtype, m=9, K=4):
    ivq = diffsptk.InverseVectorQuantization()

    tmp1 = "ivq.tmp1"
    tmp2 = "ivq.tmp2"
    U.check_compatibility(
        device,
        dtype,
        ivq,
        [f"echo 0 3 1 2 | x2x +ad > {tmp1}", f"nrand -s 234 -l {K * (m + 1)} > {tmp2}"],
        [f"cat {tmp1}", f"cat {tmp2}"],
        f"x2x +di {tmp1} | imsvq -m {m} -s {tmp2}",
        [f"rm {tmp1} {tmp2}"],
        dx=[None, m + 1],
        dy=m + 1,
    )
