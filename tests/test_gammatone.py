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

import os

import numpy as np
import pytest
import torch

import diffsptk
import tests.utils as U


@pytest.mark.parametrize("exact", [False, True])
def test_analysis(device, dtype, exact, M=4, L=8192, sr=16000, verbose=False):
    impulse = diffsptk.impulse(L - 1, device=device, dtype=dtype).view(1, 1, -1)
    gammatone = diffsptk.GammatoneFilterBankAnalysis(
        sr, filter_order=M, exact=exact, device=device
    )
    impulse_resonse = gammatone(impulse).squeeze(0)
    frequency_response = torch.fft.rfft(impulse_resonse.real, dim=-1)
    amplitude = 20 * torch.log10(torch.abs(frequency_response) + 1e-6)
    max_amplitude = torch.amax(amplitude, dim=1)
    min_amplitude = torch.amin(amplitude, dim=1)
    assert torch.allclose(
        max_amplitude,
        torch.zeros_like(max_amplitude),
        atol=1e-1,
    )
    assert torch.all(min_amplitude < -60)

    if verbose:
        suffix = "_exact" if exact else ""
        tmp = "tmp.dat"
        amplitude.cpu().numpy().astype(np.float64).tofile(tmp)
        cmd = (
            f"./tools/SPTK/tools/venv/bin/python ./tools/SPTK/bin/fdrw {tmp} "
            f"filter{suffix}.png -n {L // 2 + 1} -g -F 2 "
            f"-xname 'Frequency [Hz]' -xscale {sr / 2} "
            "-yname 'Log amplitude [dB]' "
        )
        U.call(cmd, get=False)
        os.remove(tmp)
