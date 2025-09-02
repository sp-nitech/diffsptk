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

import importlib.util
from pathlib import Path

import torch


def test_readme_examples():
    # Extract code blocks from the README file.
    with Path("README.md").open("r") as f:
        lines = f.read().splitlines()
    titles = []
    code_blocks = []
    code_block = []
    is_code_block = False
    for line in lines:
        if line.startswith("###"):
            title = line.strip()
        if is_code_block:
            if line == "```":
                if 0 < len(code_block):
                    titles.append(title)
                    code_blocks.append("\n".join(code_block))
                    code_block = []
                is_code_block = False
            else:
                code_block.append(line)
        else:
            if line == "```python":
                is_code_block = True

    assert 0 < len(code_blocks)

    # Execute the code blocks.
    print()
    for title, code_block in zip(titles, code_blocks):
        print(f"{title}")
        if "librosa" in code_block and importlib.util.find_spec("librosa") is None:
            continue
        if "cuda" in code_block and not torch.cuda.is_available():
            continue
        try:
            exec(code_block)
        except Exception:
            print(code_block)
            assert False

    # Remove the generated files.
    for file in Path(".").glob("*.wav"):
        file.unlink()
