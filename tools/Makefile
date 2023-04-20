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

all: SPTK toml

SPTK:
	git clone https://github.com/sp-nitech/SPTK.git
	cd SPTK; make

toml:
	mkdir -p toml
	wget https://github.com/gnprice/toml-cli/releases/download/v0.2.3/toml-0.2.3-x86_64-linux.tar.gz -O toml.tar.gz
	tar xzvf toml.tar.gz -C toml --strip-components 1
	rm -rf toml.tar.gz

update:
	cd SPTK; git pull; make

clean:
	rm -rf SPTK toml

.PHONY: all update clean
