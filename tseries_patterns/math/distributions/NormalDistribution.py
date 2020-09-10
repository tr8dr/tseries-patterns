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
import scipy.stats


class NormalDistribution:

    def __init__(self, mu: float, sigma: float):
        self.mu = mu
        self.sigma = sigma
        self.dist = scipy.stats.norm(mu, sigma)


    def f(self, x):
        """
        density function

        :param x: value on domain or an array of values
        :return:
        """
        return self.dist.pdf(x)


    def logf(self, x):
        """
        Log density function

        :param x: value on domain or an array of values
        :return:
        """
        return np.log(self.dist.pdf(x))


    def cum(self, x0: float, x1: float):
        """
        cumulative density function
        """
        return self.dist.cdf(x1) - self.dist.cdf(x0)

