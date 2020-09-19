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


import random
import unittest
import numpy as np
import pandas as pd

from tseries_patterns.common.utils import constrain, frange, cbind


class EmpiricalDistribution1D:
    """
    Empirically built distribution on some domain:

    1. domain [xmin, xmax]
    2. number of bins
    """

    def __init__(self, xdomain: list, xbins = 100):

        self._domain = xdomain
        self._dx = (xdomain[1] - xdomain[0]) / (xbins-1)
        self._mass = np.zeros((xbins,), dtype=float)
        self._xbins = xbins
        self._iright = -1
        self._ileft = xbins
        self._wsum = 0.0
        self._scale = 0.0


    def median (self):
        """
        Median of distribution
        """
        return self.icum (0.5)


    def mean (self):
        """
        Mean of distribution
        """
        return self._wsum / self._scale


    def toSeries(self):
        """
        Provide distribution as series
        """
        x = np.linspace(self._domain[0], self._domain[1], self._xbins)
        y = self._mass
        return pd.DataFrame (cbind(x,y), columns=['x','density'])


    def observations (self, n = 1000):
        """
        Provide a list of faked observations
        """
        observations = []
        scale = n / self._scale

        x = self._domain[0]
        for i in range(self._xbins):
            n = int(self._mass[i] * scale)
            observations += [x] * n
            x += self._dx

        return observations


    def addEvent (self, x: float, p = 1.0):
        """
        Add event / sample to distribution
        """
        x = constrain(x, self._domain[0], self._domain[1])
        rx = (x - self._domain[0]) / self._dx + 0.5
        ix = int(rx)
        aliasing = (rx - ix - 0.5)

        if aliasing >= 0:
            self.__add_mass (ix, (1 - aliasing) * p)
            self.__add_mass (ix + 1, aliasing * p)
        else:
            self.__add_mass (ix, (1 + aliasing) * p)
            self.__add_mass (ix - 1, -aliasing * p)

        self._wsum += x * p


    def addEventList (self, xlist, p = 1.0):
        """
        Add list of events to distribution
        """
        for x in xlist:
            self.addEvent(x, p)


    def addEventRange (self, x0: float, x1: float, p = 1.0):
        """
        Add event to distribution across a range
        """
        x0 = constrain(x0, self._domain[0], self._domain[1])
        x1 = constrain(x1, self._domain[0], self._domain[1])

        for x in frange(x0, x1, self._dx):
            self.addEvent (x0, p = p)


    def reset (self):
        """
        Clear distribution
        """
        self._wsum = 0.0
        self._scale = 0.0
        self._iright = -1
        self._ileft = self._xbins

        for i in range(0, self._xbins):
            self._mass[i] = 0.0


    def f(self, x: float):
        """
        determine the probability of x
        """
        if x < self._domain[0] or x > self._domain[1]:
            return 0.0

        rx = (x - self._domain[0]) / self._dx + 0.5
        ix = int(rx)
        aliasing = (rx - ix - 0.5)

        if aliasing >= 0:
            return self._mass[ix] * aliasing + self._mass[ix+1] * (1 - aliasing)
        else:
            return self._mass[ix] * (1+aliasing) + self._mass[ix-1] * -aliasing


    def cum(self, x0: float, x1 = None):
        """
        determine cumulative probability between x0 and x1 or start of distribution to x
        """
        xstart = self._domain[0]
        xend = self._domain[1]

        if x1 == None:
            x1 = constrain(x0, xstart, xend)
            x0 = xstart
        else:
            x0 = constrain(x0, xstart, xend)
            x1 = constrain(x1, xstart, xend)

        Srx = (x0 - xstart) / self._dx + 0.5
        Erx = (x1 - xstart) / self._dx + 0.5

        Si = int(Srx)
        Ei = int(Erx)

        ## main mass area
        cum = np.sum(self._mass[(Si+1):Ei])

        ## left boundary
        Laliasing = Srx - Si
        cum += ((self._mass [Si] * (1 - Laliasing)) if (Si > 0) else self._mass [Si])

        ## right boundary
        Raliasing = (Erx - Ei)
        cum += self._mass [Ei] * (1 - Raliasing)

        return cum / self._scale


    def icum(self, p: float):
        """
        Inverse cumulative function, finds x, such that f(x) = p
        """
        if self._iright < 0:
            raise Exception ("cannot compute inverse cumulative without samples")

        ## convert probability into mass
        mass = p * self._scale

        ## determine cumulative probability >= target
        cc = np.cumsum(self._mass)
        indices, = np.where (cc >= mass)
        Ci = indices[0] if len(indices) > 0 else self._iright
        Pi = Ci - 1

        Csum = cc[Ci]
        Psum = cc[Pi] if Pi >= 0 else 0.0

        xstart = self._domain[0]
        dx = self._dx
        Px = xstart + float(Pi) * dx + dx / 2

        return Px + (mass - Psum) / (Csum - Psum) * dx


    def sample (self):
        """
        Get a random sample from this distribution
        """
        r = random.uniform(0,1)
        return self.icum(r)


    @staticmethod
    def toDistribution (bins, xmin: float, xmax: float):
        """
        Create distribution from raw bins

        :param bins: vector of density bins
        :param xmin: minimum X
        :param xmax: maximum X
        :return: distribution
        """
        xbins = len(bins)
        x = np.linspace(xmin, xmax, xbins)

        dist = EmpiricalDistribution1D ([xmin, xmax], xbins)
        dist._mass = bins
        dist._scale = np.sum(bins)
        dist._wsum = np.average(x, weights=dist._mass) * dist._scale
        dist._ileft = 0
        dist._iright = xbins-1
        return dist


    #
    #   Implementation
    #

    def __add_mass (self, x: int, mass: float):
        x = constrain (x, 0, self._xbins-1)
        self._scale += mass
        self._mass[x] += mass

        if x > self._iright:
            self._iright = x
        if x < self._ileft:
            self._ileft = x




##
##  UNIT TESTS
##

class TestHistogramDistribution(unittest.TestCase):

    def setUp(self):
        self._events = [
            -0.434321819222599, -0.961912087050633, -0.555636216497691, 1.16381728550684, -0.606392963352275, -0.0414869567752631, 0.528860498027734, 0.784168847438405, -1.13892935415509, -1.7207331857625, -0.0163921341479961, -1.76232311590338, 0.10144960283406, -1.59388334584018, -0.210310745396894, -1.68960972002286, -1.59810112286596, -0.735359001954469, -0.258238284165446, -0.82844470201103, -0.31772864381216, 1.58344988860738, -0.73706528281108, 0.85869748129578, -1.12104337306646, -1.51393876778318, 0.145361555872937, 0.807008077270184, -0.817902575345228, -0.0262225492025741, 0.891160073253426, 0.311588078111408, -0.0158332283040417, 0.296994641268305, -0.100008160518654, 0.456753894052354, 0.450960099088492, 0.41229702663055, -0.588661923452689, 1.79115674400293, 1.75928443726685, 0.329293059514518, 0.306645879320997, 0.148914395831754, -0.976745767897404, -0.147880662805419, -1.50573568920172, -0.802942147177643, -0.209452720974089, -0.0193959495009254, 0.414301876259524, 0.225285478559928, 0.0562016717501315, -0.228005165863454, -0.58115460443061, -1.86770235717982, 0.054377786136995, 0.471162936316966, 0.0829002158628746, -0.920122927448515, 1.54499008615919, -0.134403967673002, -0.0877077650417071, -0.34553773697647, -0.145249230268327, -0.104275691799501, -0.129113816943174, -0.0846203281232104, -0.586331393291897, -0.421608164931027, -0.429226499504491, 0.882494948902256, 0.180090038256461, -0.414943004917648, -1.03226579145463, -0.244159745780004, 0.976669808587214, -1.29985125686636, -1.70126875947132, 0.620828836023965, 1.15751329481892, 0.669785597727839, -0.126214497338714, -1.00206544240048, -0.31256968342, -0.989098891567992, 0.125247760795736, 0.210789866141077, 2.06609325678976, -0.913350763096249, 0.23760250913037, -1.65617703794968, 0.595395922022009, 1.22322359185298, -2.00167685124865, -1.38926666933657, 0.234409567066189, -0.870163924567959, 1.86620667188135, -0.406698659106525]


    def test_mean(self):
        # setup distribution
        dist = EmpiricalDistribution1D ([-50,50], 101)

        ## add events
        for x in range(1,10+1):
            dist.addEvent(x, p=10)

        ## test mean
        self.assertAlmostEqual (5.5, dist.mean())

        ## test median
        self.assertAlmostEqual (5.5, dist.median())
        self.assertAlmostEqual (3.0, dist.icum(0.25))



    def test_mean2(self):
        # setup distribution
        dist = EmpiricalDistribution1D ([-3,3], 100)

        ## add events
        for x in self._events:
            dist.addEvent(x, p=10)

        ## test mean
        self.assertAlmostEqual (sum(self._events) / len(self._events), dist.mean())

        ## test median
        #self.assertAlmostEqual (-0.1276642, dist.median())


if __name__ == '__main__':

    unittest.main()
