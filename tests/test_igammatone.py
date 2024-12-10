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
from scipy.signal import find_peaks
import torch

import diffsptk
import tests.utils as U


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("exact", [False, True])
def test_analysis_synthesis(device, exact, M=4, L=8192, desired_delay=4, verbose=False):
    if device == "cuda" and not torch.cuda.is_available():
        return

    x, sr = diffsptk.read(
        "assets/data.wav",
        double=torch.get_default_dtype() == torch.double,
        device=device,
    )

    gammatone = diffsptk.GammatoneFilterBankAnalysis(
        sr, filter_order=M, exact=exact
    ).to(device)
    igammatone = diffsptk.GammatoneFilterBankSynthesis(
        sr, filter_order=M, exact=exact, desired_delay=desired_delay
    ).to(device)
    y = igammatone(gammatone(x.unsqueeze(0)).squeeze(0), compensate_delay=True)
    assert (x - y).abs().max() < 0.1

    if verbose:
        suffix = "_exact" if exact else ""
        diffsptk.write(f"reconst{suffix}.wav", y.squeeze(), sr)

    impulse = diffsptk.impulse(L - 1, device=device)
    reconstructed_impulse = igammatone(gammatone(impulse), compensate_delay=False)
    frequency_response = torch.fft.rfft(reconstructed_impulse.squeeze())
    amplitude = 20 * torch.log10(torch.abs(frequency_response) + 1e-6)
    amplitude = amplitude.cpu().numpy()
    peaks, _ = find_peaks(amplitude)
    assert len(peaks) == len(gammatone.center_frequencies)
    assert np.abs(amplitude[peaks[1:-1]]) == pytest.approx(0, abs=0.1)

    group_dleay = diffsptk.GroupDelay(L)(reconstructed_impulse)
    group_dleay = group_dleay.cpu().numpy() / sr * 1000
    assert group_dleay.mean() == pytest.approx(desired_delay, abs=0.1)

    if verbose:
        suffix = "_exact" if exact else ""
        tmp = "tmp.dat"

        amplitude.astype(np.float64).tofile(tmp)
        cmd = (
            f"./tools/SPTK/tools/venv/bin/python ./tools/SPTK/bin/fdrw {tmp} "
            f"amplitude{suffix}.png -g -F 2 "
            f"-xname 'Frequency [Hz]' -xscale {sr/2} "
            "-yname 'Log amplitude [dB]' -y -30 5"
        )
        U.call(cmd, get=False)

        group_dleay.astype(np.float64).tofile(tmp)
        cmd = (
            f"./tools/SPTK/tools/venv/bin/python ./tools/SPTK/bin/fdrw {tmp} "
            f"grpdelay{suffix}.png -g -F 2 "
            f"-xname 'Frequency [Hz]' -xscale {sr/2} "
            "-yname 'Group delay [ms]'"
        )
        U.call(cmd, get=False)

        os.remove(tmp)
