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

PROJECT := diffsptk
MODULE  :=

PYTHON_VERSION     := 3.9
TORCH_VERSION      := 1.11.0
TORCHAUDIO_VERSION := 0.11.0
PLATFORM           := cu113

venv:
	test -d venv || python$(PYTHON_VERSION) -m venv venv
	. ./venv/bin/activate && python -m pip install pip --upgrade
	. ./venv/bin/activate && python -m pip install torch==$(TORCH_VERSION)+$(PLATFORM) torchaudio==$(TORCHAUDIO_VERSION)+$(PLATFORM) \
		-f https://download.pytorch.org/whl/$(PLATFORM)/torch_stable.html
	. ./venv/bin/activate && python -m pip install -e .[dev]
	. ./venv/bin/activate && python -m pip install icc-rt

dist:
	. ./venv/bin/activate && python -m build
	. ./venv/bin/activate && python -m twine check dist/*

dist-clean:
	rm -rf dist

doc:
	. ./venv/bin/activate && cd docs && make html

doc-clean:
	@if [ -f ./venv/bin/activate ]; then \
		. ./venv/bin/activate && cd docs && make clean; \
	fi

check: tool
	. ./venv/bin/activate && python -m ruff check $(PROJECT) tests
	. ./venv/bin/activate && python -m ruff format --check $(PROJECT) tests
	. ./venv/bin/activate && python -m isort --check $(PROJECT) tests
	./tools/taplo/taplo fmt --check pyproject.toml
	./tools/yamlfmt/yamlfmt --lint *.yml .github/workflows/*.yml

format: tool
	. ./venv/bin/activate && python -m ruff check --fix $(PROJECT) tests
	. ./venv/bin/activate && python -m ruff format $(PROJECT) tests
	. ./venv/bin/activate && python -m isort $(PROJECT) tests
	./tools/taplo/taplo fmt pyproject.toml
	./tools/yamlfmt/yamlfmt *.yml .github/workflows/*.yml

test: tool
	[ -n "$(MODULE)" ] && module=tests/test_$(MODULE).py || module=; \
	. ./venv/bin/activate && export PATH=tools/SPTK/bin:$$PATH && python -m pytest $$module

test-clean:
	rm -rf tests/__pycache__
	rm -rf *.wav

tool:
	cd tools && make

tool-clean:
	cd tools && make clean

update: tool
	. ./venv/bin/activate && python -m pip install --upgrade pip
	@for package in $$(./tools/taplo/taplo get -f pyproject.toml project.optional-dependencies.dev); do \
		. ./venv/bin/activate && python -m pip install --upgrade $$package; \
	done

clean: dist-clean doc-clean test-clean tool-clean
	rm -rf venv
	find . -name "__pycache__" -type d | xargs rm -rf

.PHONY: venv dist dist-clean doc doc-clean check format test test-clean tool tool-clean update clean
