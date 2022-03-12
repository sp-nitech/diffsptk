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

import subprocess
import time

import numpy as np


def call(cmd, get=True):
    if get:
        res = subprocess.run(
            cmd + " | x2x +da -f %.10f",
            shell=True,
            text=True,
            stdout=subprocess.PIPE,
        )
        data = np.fromstring(res.stdout, sep="\n", dtype=np.float32)
        assert len(data) > 0, f"Failed to run command {cmd}"
        return data
    else:
        res = subprocess.run(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
        )
        return None


def lap():
    return time.process_time()
