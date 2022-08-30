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


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize(
    "operation, option",
    [
        ("a", 1),
        ("s", 1),
        ("m", 2),
        ("d", 2),
        ("r", 0.1),
        ("p", 2),
        ("l", 0.1),
        ("u", 1),
        ("ABS", None),
        ("INV", None),
        ("SQR", None),
        ("SQRT", None),
        ("LN", None),
        ("LOG2", None),
        ("LOG10", None),
        ("LOGX", 3),
        ("EXP", None),
        ("POW2", None),
        ("POW10", None),
        ("POWX", 3),
        ("SIN", None),
        ("COS", None),
        ("TAN", None),
    ],
)
def test_compatibility(device, operation, option, L=10):
    sopr = diffsptk.ScalarOperation(operation, option)

    opt = "" if option is None else option
    U.check_compatibility(
        device,
        [sopr, torch.abs],
        [],
        f"nrand -l {L} | sopr -ABS",
        f"sopr -{operation} {opt}",
        [],
    )

    U.check_differentiable(device, [sopr, torch.abs], [L])
