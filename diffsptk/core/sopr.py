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

import math

import torch
import torch.nn as nn


class ScalarOperation(nn.Module):
    """See `this page <https://sp-nitech.github.io/sptk/latest/main/sopr.html>`_
    for details.

    Parameters
    ----------
    operation : ['Addition', 'Subtraction', 'Multiplication', 'Division', 'Remainder', \
                 'Power', 'LowerBouding', 'UpperBouding', 'Absolute', 'Reciprocal', \
                 'Square', 'SquareRoot', 'NaturalLogarithm', 'Logarithm2', \
                 'Logarithm10', 'Logarithm', 'NaturalExponential', 'Exponential2', \
                 'Exponential10', 'Exponential', 'Sine', 'Coine', 'Tangent']
        Operation.

    option : float [scalar]
        Optional argument.

    """

    def __init__(self, operation, opt=None):
        super(ScalarOperation, self).__init__()

        if operation == "Addition" or operation == "a":
            self.func = lambda x: torch.add(x, opt)
        elif operation == "Subtraction" or operation == "s":
            self.func = lambda x: torch.sub(x, opt)
        elif operation == "Multiplication" or operation == "m":
            self.func = lambda x: torch.mul(x, opt)
        elif operation == "Division" or operation == "d":
            self.func = lambda x: torch.div(x, opt)
        elif operation == "Remainder" or operation == "r":
            self.func = lambda x: torch.fmod(x, opt)
        elif operation == "Power" or operation == "p":
            self.func = lambda x: torch.pow(x, opt)
        elif operation == "LowerBouding" or operation == "l":
            self.func = lambda x: torch.clip(x, min=opt)
        elif operation == "UpperBouding" or operation == "u":
            self.func = lambda x: torch.clip(x, max=opt)
        elif operation == "Absolute" or operation == "ABS":
            self.func = lambda x: torch.abs(x)
        elif operation == "Reciprocal" or operation == "INV":
            self.func = lambda x: torch.reciprocal(x)
        elif operation == "Square" or operation == "SQR":
            self.func = lambda x: torch.square(x)
        elif operation == "SquareRoot" or operation == "SQRT":
            self.func = lambda x: torch.sqrt(x)
        elif operation == "NaturalLogarithm" or operation == "LN":
            self.func = lambda x: torch.log(x)
        elif operation == "Logarithm2" or operation == "LOG2":
            self.func = lambda x: torch.log2(x)
        elif operation == "Logarithm10" or operation == "LOG10":
            self.func = lambda x: torch.log10(x)
        elif operation == "Logarithm" or operation == "LOGX":
            self.func = lambda x: torch.log(x) / math.log(opt)
        elif operation == "NaturalExponential" or operation == "EXP":
            self.func = lambda x: torch.exp(x)
        elif operation == "Exponential2" or operation == "POW2":
            self.func = lambda x: torch.pow(2, x)
        elif operation == "Exponential10" or operation == "POW10":
            self.func = lambda x: torch.pow(10, x)
        elif operation == "Exponential" or operation == "POWX":
            self.func = lambda x: torch.pow(opt, x)
        elif operation == "Sine" or operation == "SIN":
            self.func = lambda x: torch.sin(x)
        elif operation == "Coine" or operation == "COS":
            self.func = lambda x: torch.cos(x)
        elif operation == "Tangent" or operation == "TAN":
            self.func = lambda x: torch.tan(x)
        else:
            raise NotImplementedError(f"Operation {operation} is not supported")

    def forward(self, x):
        """Perform scalar operation.

        Parameters
        ----------
        x : Tensor [shape=(...,)]
            Input.

        Returns
        -------
        y : Tensor [shape=(...,)]
            Output.

        Examples
        --------
        >>> x = diffsptk.ramp(4)
        >>> sopr = diffsptk.ScalarOperation("Multiplication", 2)
        >>> y = sopr(x)
        >>> y
        tensor([0., 2., 4., 6., 8.])

        """
        y = self.func(x)
        return y
