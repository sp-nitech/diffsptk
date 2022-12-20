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

import filecmp
import os

import pytest

import diffsptk


@pytest.mark.parametrize("mode", ["hts", "auto"])
def test_get_alpha(mode):
    alpha = diffsptk.get_alpha(sr=48000, mode=mode)
    assert alpha == 0.55


@pytest.mark.parametrize("double", [False, True])
def test_read_and_write(double):
    in_wav = "assets/data.wav"
    out_wav = "data.wav"
    x, sr = diffsptk.read(in_wav, double=double)
    diffsptk.write(out_wav, x, sr)
    assert filecmp.cmp(in_wav, out_wav, shallow=False)
    os.remove(out_wav)
