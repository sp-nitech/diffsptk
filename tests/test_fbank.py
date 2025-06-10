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

import pytest

import diffsptk
import tests.utils as U


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("module", [False, True])
@pytest.mark.parametrize("out_format", [0, 1])
def test_compatibility(
    device, module, out_format, C=10, L=32, sr=8000, f_min=300, f_max=3400, floor=1, B=2
):
    spec = diffsptk.Spectrum(L, eps=0)
    fbank = U.choice(
        module,
        diffsptk.MelFilterBankAnalysis,
        diffsptk.functional.fbank,
        {
            "fft_length": L,
            "n_channel": C,
            "sample_rate": sr,
            "f_min": f_min,
            "f_max": f_max,
            "floor": floor,
            "out_format": out_format,
        },
    )

    s = sr // 1000
    U.check_compatibility(
        device,
        [fbank, spec],
        [],
        f"nrand -l {B * L}",
        f"fbank -n {C} -l {L} -s {s} -L {f_min} -H {f_max} -e {floor} -o {out_format}",
        [],
        dx=L,
        dy=C + out_format,
    )

    U.check_differentiability(device, [fbank, spec], [B, L])


@pytest.mark.parametrize("scale", ["htk", "mel", "inverted-mel", "bark", "linear"])
@pytest.mark.parametrize("erb_factor", [None, 0.5])
def test_analysis(scale, erb_factor, C=40, L=2048, sr=8000, verbose=False):
    fbank = diffsptk.MelFilterBankAnalysis(
        fft_length=L,
        n_channel=C,
        sample_rate=sr,
        scale=scale,
        erb_factor=erb_factor,
    )

    if verbose:
        suffix = "_e" if erb_factor is not None else ""
        tmp = "tmp.dat"
        fbank.H.T.numpy().astype("float64").tofile(tmp)
        cmd = (
            f"./tools/SPTK/tools/venv/bin/python ./tools/SPTK/bin/fdrw {tmp} "
            f"fbank_{scale}{suffix}.png -n {L // 2 + 1} -g -H 400 -W 1000 "
            f"-xname 'Frequency [Hz]' -xscale {sr / 2} "
            "-yname 'Amplitude' "
        )
        U.call(cmd, get=False)
        os.remove(tmp)


def test_learnable(C=10, L=32, sr=8000):
    fbank = diffsptk.MelFilterBankAnalysis(
        fft_length=L,
        n_channel=C,
        sample_rate=sr,
        learnable=True,
    )
    U.check_learnable(fbank, (L // 2 + 1,))
