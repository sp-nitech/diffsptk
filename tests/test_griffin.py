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
import torchaudio

import diffsptk
import tests.utils as U


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("module", [False, True])
@pytest.mark.parametrize("init_phase", ["zeros", "random"])
def test_compatibility(
    device,
    module,
    init_phase,
    n_iter=64,
    alpha=0.99,
    beta=0.99,
    gamma=1.1,
    verbose=False,
):
    if device == "cuda" and not torch.cuda.is_available():
        return

    x, sr = diffsptk.read("assets/data.wav", device=device)

    stft_params = {
        "frame_length": 512,
        "frame_period": 64,
        "fft_length": 512,
        "center": True,
        "mode": "constant",
        "window": "blackman",
        "norm": "power",
        "symmetric": True,
    }

    stft = diffsptk.STFT(**stft_params, out_format="power").to(device)
    X = stft(x)

    griffin = U.choice(
        module,
        diffsptk.GriffinLim,
        diffsptk.functional.griffin,
        {
            **stft_params,
            "n_iter": n_iter,
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
            "init_phase": init_phase,
            "verbose": verbose,
        },
    )
    if hasattr(griffin, "to"):
        griffin = griffin.to(device)
    T = x.size(0)
    y = griffin(X, out_length=T)

    if verbose:
        z = torchaudio.functional.griffinlim(
            specgram=X.transpose(-2, -1),
            window=stft.layers[1].window,
            n_fft=stft_params["fft_length"],
            hop_length=stft_params["frame_period"],
            win_length=stft_params["frame_length"],
            power=2,
            n_iter=n_iter,
            momentum=alpha,
            length=None,
            rand_init=init_phase == "random",
        )
        diffsptk.write("reconst.wav", y, sr)
        diffsptk.write("reconst_torchaudio.wav", z, sr)

    U.check_differentiability(device, [griffin, stft], [T // 10])
