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

import tests.utils as U
from diffsptk.utils.private import auditory_to_hz, hz_to_auditory


@pytest.mark.parametrize("scale", ["htk", "oshaughnessy", "traunmuller", "linear"])
def test_hz_to_auditory(scale):
    f = np.linspace(0, 8000, 100)
    g = hz_to_auditory(auditory_to_hz(f, scale), scale)
    assert U.allclose(f, g)
