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

import torch
import torch.nn as nn


class CepstrumToImpulseResponse(nn.Module):
    def __init__(self, cep_order, impulse_response_length):
        """Initialize module.

        Parameters
        ----------
        cep_order : int >= 1 [scalar]
            Order of cepstrum, M.

        impulse_response_length : int >= 1 [scalar]
            Length of impulse response, N.

        """
        super(CepstrumToImpulseResponse, self).__init__()

        self.cep_order = cep_order
        self.impulse_response_length = impulse_response_length

        assert 1 <= self.cep_order
        assert 1 <= self.impulse_response_length

        self.register_buffer("arange", torch.arange(1, self.cep_order + 1))

    def forward(self, c):
        """Convert cepstrum to impulse response.

        Parameters
        ----------
        c : Tensor [shape=(B, M+1)]
            Cesptral coefficients.

        Returns
        -------
        h : Tensor [shape=(B, N)]
            Truncated impulse response.

        """
        c0 = c[:, 0]
        c1 = c[:, 1:] * self.arange
        c1 = torch.flip(c1, [1])

        h = torch.empty((c.shape[0], self.impulse_response_length), device=c.device)
        h[:, 0] = torch.exp(c0)
        for n in range(1, self.impulse_response_length):
            s = n - self.cep_order
            h[:, n] = (
                torch.einsum(
                    "bd,bd->b",
                    h[:, max(0, s) : n].clone(),
                    c1[:, max(0, -s) :],
                )
                / n
            )
        return h
