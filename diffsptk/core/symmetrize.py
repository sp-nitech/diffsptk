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


class Symmetrization(nn.Module):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/symmetrize.html>`_
    for details.

    Parameters
    ----------
    in_format : ['0H', '0H1', 'H0H/2', 'H0H']
        Input format.

    out_format : ['0H', '0H1', 'H0H/2', 'H0H']
        Output format.

    """

    def __init__(self, in_format="0H", out_format="0H1"):
        super(Symmetrization, self).__init__()

        if in_format == 0 or in_format == "0H":
            self.in_format = "0H"
        elif in_format == 1 or in_format == "0H1":
            self.in_format = "0H1"
        elif in_format == 2 or in_format == "H0H/2":
            self.in_format = "H0H/2"
        elif in_format == 3 or in_format == "H0H":
            self.in_format = "H0H"
        else:
            raise ValueError(f"input format {in_format} is not supported")

        if out_format == 0 or out_format == "0H":
            self.out_format = "0H"
        elif out_format == 1 or out_format == "0H1":
            self.out_format = "0H1"
        elif out_format == 2 or out_format == "H0H/2":
            self.out_format = "H0H/2"
        elif out_format == 3 or out_format == "H0H":
            self.out_format = "H0H"
        else:
            raise ValueError(f"output format {out_format} is not supported")

    def forward(self, x):
        """Symmetrize data.

        Parameters
        ----------
        x : Tensor [shape=(..., ?)]
            FFT output.

        Returns
        -------
        y : Tensor [shape=(..., ?)]
            Symmetrized output.

        Examples
        --------
        >>> x = diffsptk.ramp(3)
        >>> x
        tensor([0., 1., 2., 3.])
        >>> symmetrize = diffsptk.Symmetrization(out_format="0H1")
        >>> y = symmetrize(x)
        >>> y
        tensor([0., 1., 2., 3., 2., 1.])
        >>> symmetrize = diffsptk.Symmetrization(out_format="H0H")
        >>> y = symmetrize(x)
        >>> y
        tensor([3., 2., 1., 0., 1., 2., 3.])

        """
        H = x.size(-1) // 2

        if self.in_format == "0H":
            if self.out_format == "0H":
                y = x
            elif self.out_format == "0H1":
                y = torch.cat((x, x[..., 1:-1].flip(-1)), dim=-1)
            elif self.out_format == "H0H/2":
                edge = 0.5 * x[..., -1:]
                y = torch.cat((edge, x[..., 1:-1].flip(-1), x[..., :-1], edge), dim=-1)
            elif self.out_format == "H0H":
                y = torch.cat((x[..., 1:].flip(-1), x), dim=-1)
            else:
                raise RuntimeError
        elif self.in_format == "0H1":
            H1 = H + 1
            if self.out_format == "0H":
                y = x[..., :H1]
            elif self.out_format == "0H1":
                y = x
            elif self.out_format == "H0H/2":
                edge = 0.5 * x[..., H:H1]
                y = torch.cat((edge, x[..., H1:], x[..., :H], edge), dim=-1)
            elif self.out_format == "H0H":
                y = torch.cat((x[..., H:], x[..., :H1]), dim=-1)
            else:
                raise RuntimeError
        elif self.in_format == "H0H/2":
            if self.out_format == "0H":
                y = torch.cat((x[..., H:-1], 2 * x[..., -1:]), dim=-1)
            elif self.out_format == "0H1":
                y = torch.cat((x[..., H:-1], 2 * x[..., -1:], x[..., 1:H]), dim=-1)
            elif self.out_format == "H0H/2":
                y = x
            elif self.out_format == "H0H":
                edge = 2 * x[..., -1:]
                y = torch.cat((edge, x[..., 1:-1], edge), dim=-1)
            else:
                raise RuntimeError
        elif self.in_format == "H0H":
            if self.out_format == "0H":
                y = x[..., H:]
            elif self.out_format == "0H1":
                y = torch.cat((x[..., H:], x[..., 1:H]), dim=-1)
            elif self.out_format == "H0H/2":
                edge = 0.5 * x[..., -1:]
                y = torch.cat((edge, x[..., 1:-1], edge), dim=-1)
            elif self.out_format == "H0H":
                y = x
            else:
                raise RuntimeError
        else:
            raise RuntimeError

        return y
