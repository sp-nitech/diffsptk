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

PROJECT        := diffsptk
PYTHON_VERSION := 3.8
MODULE         :=

init:
	pip install -e .

dev:
	test -d venv || python$(PYTHON_VERSION) -m venv venv; \
	. ./venv/bin/activate; pip install pip --upgrade; \
	pip install torch==1.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html; \
	pip install -e .[dev]

dist:
	. ./venv/bin/activate; python -m build --wheel; \
	twine check dist/*

dist-clean:
	rm -rf dist

doc:
	. ./venv/bin/activate; cd docs; make html

doc-clean:
	@if [ -f ./venv/bin/activate ]; then \
		. ./venv/bin/activate; cd docs; make clean; \
	fi

check:
	./venv/bin/black --check $(PROJECT) tests
	./venv/bin/isort --check $(PROJECT) tests --project $(PROJECT)
	./venv/bin/pflake8 $(PROJECT) tests

format:
	./venv/bin/black $(PROJECT) tests
	./venv/bin/isort $(PROJECT) tests --project $(PROJECT)
	./venv/bin/pflake8 $(PROJECT) tests

test:
	@if [ ! -d tools/SPTK/bin ]; then \
		echo ""; \
		echo "Error: please install C++ version of SPTK"; \
		echo ""; \
		echo "  make tool"; \
		echo ""; \
		exit 1; \
	fi
	[ -n "$(MODULE)" ] && module=tests/test_$(MODULE).py || module=; \
	. ./venv/bin/activate; export PATH=tools/SPTK/bin:$$PATH; \
		python -m pytest -s --cov=./ --cov-report=xml $$module

test-clean:
	rm -rf tests/__pycache__
	rm -rf *.wav

tool:
	cd tools; make

tool-clean:
	cd tools; make clean

update:
	@if [ ! -x tools/toml/toml ]; then \
		echo ""; \
		echo "Error: please install toml-cli"; \
		echo ""; \
		echo "  make tool"; \
		echo ""; \
		exit 1; \
	fi
	./venv/bin/python -m pip install --upgrade pip
	@for package in $$(./tools/toml/toml get pyproject.toml project.optional-dependencies.dev | \
		sed 's/"//g' | tr -d '[]' | tr , ' '); do \
		./venv/bin/pip install --upgrade $$package; \
	done

clean: dist-clean doc-clean test-clean tool-clean
	rm -rf venv
	find . -name "__pycache__" -type d | xargs rm -rf

.PHONY: init dev dist dist-clean doc doc-clean check format test test-clean tool tool-clean update clean
