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


class LaplaceDistribution:

    def __init__(self, mu: float, beta: float):
        self.mu = mu
        self.beta = beta


    def f(self, x):
        """
        density function

        :param x: value on domain or an array of values
        :return:
        """
        return 1 / (2*self.beta) * np.exp(-np.abs(x - self.mu) / self.beta)


    def logf(self, x):
        """
        Log density function

        :param x: value on domain or an array of values
        :return:
        """
        beta = self.beta
        return -(np.abs(x - self.mu) + beta * np.log(2 * beta)) / beta


    def cum(self, x0: float, x1: float):
        """
        cumulative density function
        """
        mu = self.mu
        beta = self.beta

        if x0 <= self.mu:
            cdf1 = 0.5 * np.exp((x0 - mu) / beta)
        else:
            cdf1 = 1 - 0.5 * np.exp(-(x0 - mu) / beta)

        if x1 <= self.mu:
            cdf1 = 0.5 * np.exp((x1 - mu) / beta)
        else:
            cdf1 = 1 - 0.5 * np.exp(-(x1 - mu) / beta)

