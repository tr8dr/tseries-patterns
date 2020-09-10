#
# MIT License
#
# Copyright (c) 2018 Jonathan Shore
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#

import numpy as np


class ExponentialDistribution:

    def __init__(self, base: float, decay: float, dir=1.0):
        self.base = base
        self.decay = decay
        self.dir = dir


    def f(self, x):
        """
        density function

        :param x: value on domain or an array of values
        :return:
        """
        dx = self.dir * (x - self.base)
        return self.decay * np.exp(-self.decay * dx)


    def logf(self, x):
        """
        Log density function

        :param x: value on domain or an array of values
        :return:
        """
        dx = self.dir * (x - self.base)
        return np.log(self.decay) - self.decay * dx


    def cum(self, x0: float, x1: float):
        """
        cumulative density function
        """
        x0 = self.dir * (x0 - self.base)
        x1 = self.dir * (x1 - self.base)

        cdf0 = 1 - self.decay * np.exp(-self.decay * x0)
        cdf1 = 1 - self.decay * np.exp(-self.decay * x1)
        return cdf1 - cdf0
