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
@pytest.mark.parametrize("w", [0, 1, 2, 3, 4, 5, 6])
@pytest.mark.parametrize("norm", [0, 1, 2])
@pytest.mark.parametrize("symmetric", [False, True])
@pytest.mark.parametrize("L1", [8, 10])
def test_compatibility(device, module, w, norm, symmetric, L1, L2=10, B=2):
    window = U.choice(
        module,
        diffsptk.Window,
        diffsptk.functional.window,
        {
            "in_length": L1,
            "out_length": L2,
            "window": w,
            "symmetric": symmetric,
            "norm": norm,
        },
    )

    opt = "" if symmetric else "-p"
    U.check_compatibility(
        device,
        window,
        [],
        f"step -l {L1}",
        f"window -w {w} -n {norm} -l {L1} -L {L2} {opt}",
        [],
        dx=L1,
        dy=L2,
    )

    U.check_differentiability(device, window, [B, L1])


def test_povey_window(device="cpu", L=8):
    window = diffsptk.Window(L, window="povey", norm="none")

    U.check_compatibility(
        device,
        window,
        [],
        f"step -l {L}",
        f"window -w 2 -n -0 -l {L} -L {L} | sopr -p 0.85",
        [],
        dx=L,
        dy=L,
    )


def test_learnable(L=8):
    window = diffsptk.Window(L, learnable=True)
    U.check_learnable(window, (L,))
