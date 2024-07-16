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

TAPLO_VERSION   := 0.9.2
YAMLFMT_VERSION := 0.13.0

all: SPTK taplo yamlfmt

SPTK:
	git clone https://github.com/sp-nitech/SPTK.git
	cd SPTK && make

taplo:
	mkdir -p taplo
	wget https://github.com/tamasfe/taplo/releases/download/$(TAPLO_VERSION)/taplo-linux-x86_64.gz -O taplo.gz
	gunzip -c taplo.gz > taplo/taplo
	chmod +x taplo/taplo
	rm -f taplo.gz

yamlfmt:
	mkdir -p yamlfmt
	wget https://github.com/google/yamlfmt/releases/download/v$(YAMLFMT_VERSION)/yamlfmt_$(YAMLFMT_VERSION)_Linux_x86_64.tar.gz -O yamlfmt.tar.gz
	tar xzf yamlfmt.tar.gz -C yamlfmt
	rm -f yamlfmt.tar.gz

clean:
	rm -rf SPTK taplo yamlfmt

.PHONY: all clean
