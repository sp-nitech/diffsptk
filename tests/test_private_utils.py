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
import torch

import tests.utils as U
from diffsptk.utils.private import auditory_to_hz, fill_warping_matrix, hz_to_auditory


@pytest.mark.parametrize("scale", ["htk", "mel", "inverted-mel", "bark", "linear"])
def test_hz_to_auditory(scale):
    f = np.linspace(0, 8000, 100)
    g = hz_to_auditory(auditory_to_hz(f, scale), scale)
    assert U.allclose(f, g)


@pytest.mark.parametrize("alpha", [-0.5, 0, 0.42, 0.77])
@pytest.mark.parametrize("start", [1, 2])
@pytest.mark.parametrize("shape", [(1, 1), (2, 5), (5, 2), (10, 10)])
def test_fill_warping_matrix(alpha, start, shape):
    A = torch.zeros(shape, dtype=torch.double)
    A[:start] = torch.randn(min(start, shape[0]), shape[1], dtype=torch.double)
    A[:, 0] = torch.randn(shape[0], dtype=torch.double)

    target = A.clone()
    for i in range(start, shape[0]):
        for j in range(1, shape[1]):
            target[i, j] = target[i - 1, j - 1] + alpha * (
                target[i, j - 1] - target[i - 1, j]
            )

    output = fill_warping_matrix(A, alpha, start)
    assert U.allclose(target, output)
