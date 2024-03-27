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
@pytest.mark.parametrize("module", [False, True])
@pytest.mark.parametrize("norm", [False, True])
def test_compatibility(
    device, module, norm, K=4, lower_bound=-1, upper_bound=1, L=50, B=2
):
    histogram = U.choice(
        module,
        diffsptk.Histogram,
        diffsptk.functional.histogram,
        {},
        {
            "n_bin": K,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "norm": norm,
            "softness": 1e-4,
        },
    )

    opt = "-n" if norm else ""
    U.check_compatibility(
        device,
        histogram,
        [],
        [f"nrand -l {B*L}"],
        f"histogram -t {L} -b {K} -l {lower_bound} -u {upper_bound} {opt}",
        [],
        dx=L,
        dy=K,
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_differentiability(device, K=4, L=50):
    histogram = diffsptk.Histogram(K, lower_bound=-1, upper_bound=1, softness=1e-1)
    U.check_differentiability(device, histogram, [L])
