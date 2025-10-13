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
OPT     :=

PYTHON_VERSION     := 3.11
TORCH_VERSION      := 2.3.1
TORCHAUDIO_VERSION := 2.3.1
PLATFORM           := cu121

venv:
	test -d .venv || python$(PYTHON_VERSION) -m venv .venv
	. .venv/bin/activate && python -m pip install --upgrade pip
	. .venv/bin/activate && python -m pip install --upgrade wheel
	. .venv/bin/activate && python -m pip install torch==$(TORCH_VERSION)+$(PLATFORM) torchaudio==$(TORCHAUDIO_VERSION)+$(PLATFORM) \
		--index-url https://download.pytorch.org/whl/$(PLATFORM)
	. .venv/bin/activate && python -m pip install -e .[dev]

dist:
	. .venv/bin/activate && python -m build
	. .venv/bin/activate && python -m twine check dist/*

dist-clean:
	rm -rf dist

doc:
	. .venv/bin/activate && cd docs && make html

doc-clean:
	@if [ -f .venv/bin/activate ]; then \
		. .venv/bin/activate && cd docs && make clean; \
	fi

check: tool
	. .venv/bin/activate && python -m ruff check $(PROJECT) tests
	. .venv/bin/activate && python -m ruff format --check $(PROJECT) tests docs/source
	. .venv/bin/activate && python -m mdformat --check *.md
	.venv/bin/codespell
	./tools/taplo/taplo fmt --check *.toml
	./tools/yamlfmt/yamlfmt --lint *.cff *.yml .github/workflows/*.yml

format: tool
	. .venv/bin/activate && python -m ruff check --fix $(PROJECT) tests
	. .venv/bin/activate && python -m ruff format $(PROJECT) tests docs/source
	. .venv/bin/activate && python -m mdformat *.md
	./tools/taplo/taplo fmt *.toml
	./tools/yamlfmt/yamlfmt *.cff *.yml .github/workflows/*.yml

test-all: test-example test

test: tool
	[ -n "$(MODULE)" ] && module="--no-cov tests/test_$(MODULE).py" || module=; \
	. .venv/bin/activate && export PATH=./tools/SPTK/bin:$$PATH NUMBA_CUDA_LOW_OCCUPANCY_WARNINGS=0 PYLSTRAIGHT_DEBUG=1 && \
	python -m pytest $$module $(OPT)

test-example: tool
	[ -n "$(MODULE)" ] && module=modules/$(MODULE).py || module=; \
	. .venv/bin/activate && export NUMBA_CUDA_LOW_OCCUPANCY_WARNINGS=0 && \
	python -m pytest --doctest-modules --no-cov --ignore=diffsptk/third_party diffsptk/$$module

test-clean:
	rm -rf tests/__pycache__
	rm -rf *.png *.wav

tool:
	cd tools && make

tool-clean:
	cd tools && make clean

update: tool
	. .venv/bin/activate && python -m pip install --upgrade pip
	@for package in $$(./tools/taplo/taplo get -f pyproject.toml project.optional-dependencies.dev); do \
		. .venv/bin/activate && python -m pip install --upgrade $$package; \
	done

clean: dist-clean doc-clean test-clean tool-clean
	rm -rf .venv
	find . -name "__pycache__" -type d | xargs rm -rf

.PHONY: venv dist dist-clean doc doc-clean check format test test-clean tool tool-clean update clean
