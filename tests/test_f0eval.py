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
@pytest.mark.parametrize(
    "out_format",
    [
        "f0-rmse-hz",
        "f0-rmse-cent",
        "f0-rmse-semitone",
        "vuv-error-rate",
        "vuv-error-percent",
    ],
)
@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
def test_compatibility(device, dtype, module, reduction, out_format, B=2, L=10):
    f0eval = U.choice(
        module,
        diffsptk.F0Evaluation,
        diffsptk.functional.f0eval,
        {"reduction": reduction, "out_format": out_format},
    )

    tmp1 = "f0eval.tmp1"
    tmp2 = "f0eval.tmp2"
    if out_format == "f0-rmse-hz":
        cmd = f"rmse -magic 0 {tmp1} {tmp2}"
    else:
        o = 1 if out_format.startswith("f0-rmse") else 2
        mul = 0.01 if out_format in ("f0-rmse-semitone", "vuv-error-rate") else 1
        cmd = f"f0eval -q 1 -o {o} {tmp1} {tmp2} | sopr -m {mul}"

    U.check_compatibility(
        device,
        dtype,
        f0eval,
        [
            f"echo 0 0 200 210 0 200 0 | x2x +ad > {tmp1}",
            f"echo 0 0 190 180 180 0 0 | x2x +ad > {tmp2}",
        ],
        [f"cat {tmp1}", f"cat {tmp2}"],
        cmd,
        [f"rm {tmp1} {tmp2}"],
    )

    if out_format.startswith("f0-rmse"):
        U.check_differentiability(
            device, dtype, f0eval, [(B, L), (B, L)], nonnegative_input=True
        )


def test_f1_score():
    f0eval = diffsptk.F0Evaluation(out_format="vuv-macro-f1-score")
    x = torch.tensor([0, 1, 1, 0, 0, 1, 0, 1, 0])
    y = torch.tensor([0, 1, 0, 0, 1, 0, 0, 1, 1])
    f1_score = f0eval(x, y)
    assert U.allclose(f1_score, torch.tensor(0.55))
