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

init:
	pip install -e .

dev:
	@if type virtualenv > /dev/null 2>&1; then \
		test -d venv || virtualenv -p python$(PYTHON_VERSION) venv; \
	else \
		test -d venv || python$(PYTHON_VERSION) -m venv venv; \
	fi
	. venv/bin/activate; pip install pip --upgrade; pip install -e .[dev]

dist:
	./venv/bin/python setup.py bdist_wheel
	./venv/bin/twine check dist/*

dist-clean:
	@if [ -f ./venv/bin/python ]; then \
		./venv/bin/python setup.py clean --all; \
	fi
	rm -rf dist

doc:
	. venv/bin/activate; cd docs; make html

doc-clean:
	@if [ -f ./venv/bin/activate ]; then \
		. venv/bin/activate; cd docs; make clean; \
	fi

format:
	./venv/bin/black $(PROJECT) tests
	./venv/bin/isort $(PROJECT) tests --project $(PROJECT)
	./venv/bin/flake8 $(PROJECT) tests --exclude __init__.py

test:
	. venv/bin/activate; export PATH=tools/SPTK/bin:$$PATH; \
		pytest -s --cov=./ --cov-report=xml

tool:
	cd tools; make

tool-clean:
	cd tools; make clean

update:
	@for package in $$(cat setup.py | grep "           " | sed "s/\s//g" | \
	sed 's/"//g' | tr ",\n" " "); do \
		./venv/bin/pip install --upgrade $$package; \
	done

clean: dist-clean doc-clean tool-clean
	rm -rf *.egg-info venv
	find . -name "__pycache__" -type d | xargs rm -rf

.PHONY: init dev dist dist-clean doc doc-clean format test tool tool-clean update clean
