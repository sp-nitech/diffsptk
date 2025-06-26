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


def pytest_addoption(parser):
    parser.addoption(
        "--only_float",
        action="store_true",
        default=False,
        help="Check only float dtype.",
    )


@pytest.fixture(scope="session", autouse=True)
def set_default_dtype(request):
    pytest.only_float = request.config.getoption("--only_float")


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("Skipping CUDA tests as no GPU is available.")
    return request.param


@pytest.fixture(params=[None, torch.double])
def dtype(request):
    if request.param is None and request.node.get_closest_marker("skip_float_check"):
        pytest.skip("Skipping float dtype check.")
    if request.param == torch.double and pytest.only_float:
        pytest.skip("Skipping double dtype check.")
    return request.param
