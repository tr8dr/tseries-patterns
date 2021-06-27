#
# MIT License
#
# Copyright (c) 2015 Jonathan Shore
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


import pandas as pd
import numpy as np

import plotnine
from plotnine import *

from tseries_patterns.common import PriceType
from tseries_patterns.common.rendering import scale_x_datetime_auto
from tseries_patterns.common.utils import columnFor


cdef int max (int a, int b):
    if a > b:
        return a
    else:
        return b


cdef class AmplitudeBasedLabeler:
    """
    Labels upward and downward momentum (or trend) movements where the following criteria are observed:
    - movement amplitude > minamp (usually defined in bps)
    - movement makes a new high (low) within Tinactive samples (where samples are # of bars)
    - movement is not broken by a move of minamp in the opposite direction

    These simple rules, together with a least squares filtration method, work astonishingly well in accurately
    identifying moves.

    It is recommended that prices be converted into cumulative returns, such that `minamp` can be defined
    from a return (or basis point) perspective.  Usually this would be accomplished by applying the following
    to a price series (in this case converting into cumulative bps):

    cumr = np.log(prices / prices[0]) * 1e4

    Amplitude `minamp` can then be defined as, say, 25bps instead of amplitude in price terms.
    """

    cdef double minamp
    cdef int Tinactive
    cdef object df

    def __init__(self, minamp, Tinactive):
        """
        Label upward and downward momentum (or trend) movements

        :param minamp: minimum amplitude of move (usually in bps)
        :param Tinactive: maximum inactive period where no new high (low) achieved (unit: # of samples)
        """
        self.minamp = minamp
        self.Tinactive = Tinactive
        self.df = None


    def label (self, prices, type = PriceType.PRICE, scale = 1e4):
        """
        Perform labeling

        :param prices: vector of bars, prices, or cumulative returns
        :param type: indicates whether in price, cumulative BPS, or cumulative return form
        :return: labels for the series
        """
        if isinstance(prices, pd.DataFrame):
            prices = prices.reset_index()
            times = columnFor (prices, ["time", "date", "Date","Datetime", "stamp"])
            prices = columnFor (prices, ["Adj Close", "Close", "close", "price"])
        else:
            prices = pd.Series(prices)
            times = np.array(0,prices.shape[0])

        cumr = type.toBps(prices, scale = scale)
        n = cumr.shape[0]
        labels = np.zeros(n).astype(np.double)

        self._pass1 (cumr, labels)
        self._filter (cumr, labels)
        self.df = pd.DataFrame({'stamp': times, 'price': cumr, 'label': labels})
        return self.df


    def plot(
        self,
        color_price = 'darkgray',
        colors_dir = ["red", "lightgrey", "#10a4f4"],
        pointsize = 1.0,
        figsize = (10,8),
        title = ""):
        """
        Plot price (cumulative return) and labels.  Makes use of prior call to label function

        :param color_price: color for price series (dark gray default)
        :param colors_dir: colors for downward and upward momentum labels respectively
        :param pointsize: point size of label dots
        :param figsize: size of graph
        :param title: title associated with graph
        :return:
        """
        labels = self.df["label"]
        up = (labels > 0.0)
        neutral = (labels == 0.0)
        down = (-labels > 0.0)

        df1 = pd.DataFrame({'stamp': self.df["stamp"], 'price': self.df["price"]})

        plotnine.options.figure_size = figsize
        v = (ggplot() +
            geom_line(aes(x='stamp',y='price'), data=df1, color=color_price) +
            geom_point(aes(x='stamp',y='price'), data=df1.loc[down], color=colors_dir[0], size=pointsize) +
            geom_point(aes(x='stamp',y='price'), data=df1.loc[neutral], color=colors_dir[1], size=pointsize) +
            geom_point(aes(x='stamp',y='price'), data=df1.loc[up], color=colors_dir[2], size=pointsize) +
            scale_x_datetime_auto (df1["stamp"], figsize) +
            labs(title=title))

        return v


    cdef void _pass1 (self, double[:] cumr, double[:] labels):
        """
        Brute-force labeling according to minamp and Tinactive rules.  This needs to be further filtered with 
        OLS pass
        
        This code is ugly due to restrictions imposed by cython in terms of variable pre-declaration, etc.
        """

        cdef int len = cumr.shape[0]
        if len == 0:
            return

        cdef int Istart = 0
        cdef int Icursor = 0

        cdef int Imin = 0
        cdef int Imax = 0

        cdef double Vmin = cumr[0]
        cdef double Vmax = cumr[0]
        cdef double Vprior = cumr[0]

        cdef v = 0.0

        while Icursor < len:
            v = cumr[Icursor]

            # determine whether there has been a retracement, requiring a split
            if (Vmax - Vmin) >= self.minamp and Imin > Imax and (v - Vmin) >= self.minamp:
                self._apply_label (labels, Istart, Imax-1, 0.0)
                self._apply_label (labels, Imax, Imin, -1.0)
                Istart = Imin
                Imax = Icursor
                Vmax = v
            elif (Vmax - Vmin) >= self.minamp and Imax > Imin and (Vmax - v) >= self.minamp:
                self._apply_label (labels, Istart, Imin-1, 0.0)
                self._apply_label (labels, Imin, Imax, +1.0)
                Istart = Imax
                Imin = Icursor
                Vmin = v

            # check for "inactive" period where price has not progressed since latest min/max (upward direction)
            elif Imax > Imin and (Icursor - Imax) >= self.Tinactive and v <= Vmax:
                if (Vmax - Vmin) >= self.minamp:
                    self._apply_label (labels, Istart, Imin-1, 0.0)
                    self._apply_label (labels, Imin, Imax, +1.0)
                    self._apply_label (labels, Imax+1, Icursor, 0.0)
                else:
                    self._apply_label (labels, Istart, Icursor, 0.0)

                Istart = Icursor
                Imax = Icursor
                Imin = Icursor
                Vmax = v
                Vmin = v

            # check for "inactive" period where price has not progressed since latest min/max (downward direction)
            elif Imin > Imax and (Icursor - Imin) >= self.Tinactive and v >= Vmin:
                if (Vmax - Vmin) >= self.minamp:
                    self._apply_label (labels, Istart, Imax-1, 0.0)
                    self._apply_label (labels, Imax, Imin, -1.0)
                    self._apply_label (labels, Imin+1, Icursor, 0.0)
                else:
                    self._apply_label (labels, Istart, Icursor, 0.0)

                Istart = Icursor
                Imax = Icursor
                Imin = Icursor
                Vmax = v
                Vmin = v

            # adjust local maximum
            if v >= Vmax:
                Imax = Icursor
                Vmax = v
            # adjust local minimum
            if v <= Vmin:
                Imin = Icursor
                Vmin = v

            Icursor += 1

        # finish end
        if (Vmax - Vmin) >= self.minamp and Imin > Imax:
            self._apply_label (labels, Istart, Imax-1, 0.0)
            self._apply_label (labels, Imax, Imin, -1.0)
            self._apply_label (labels, Imin+1, Icursor-1, 0.0)

        elif (Vmax - Vmin) >= self.minamp and Imax > Imin:
            self._apply_label (labels, Istart, Imin-1, 0.0)
            self._apply_label (labels, Imin, Imax, +1.0)
            self._apply_label (labels, Imax+1, Icursor-1, 0.0)
        else:
            self._apply_label (labels, Istart, Icursor-1, 0.0)


    cdef void _filter (self, double[:] cumr, double[:] labels):
        """
        Using distance from OLS regression, determine which points in a momentum region belong
        
        This code is ugly due to restrictions imposed by cython in terms of variable pre-declaration, etc.
        """

        cdef int len = cumr.shape[0]
        cdef int Ipos = 0
        cdef int Istart = 0
        cdef int Iend = 0

        cdef int Imaxfwd = 0
        cdef int Imaxback = 0
        cdef double Vmaxfwd = 0.0
        cdef double Vmaxback = 0.0

        cdef double fExy = 0.0
        cdef double fExx = 0.0
        cdef double fEx = 0.0
        cdef double fEy = 0.0

        cdef double bExy = 0.0
        cdef double bExx = 0.0
        cdef double bEx = 0.0
        cdef double bEy = 0.0

        cdef double beta = 0.0
        cdef double distance = 0.0

        cdef double Xc = 0.0
        cdef double Yc = 0.0
        cdef double dir = 0.0
        cdef int i = 0

        while Ipos < len:
            dir = labels[Ipos]
            if dir == 0.0:
                Ipos += 1
                continue

            # locate end of region
            Istart = Ipos
            Iend = Ipos
            while Iend < len and labels[Iend] == dir: Iend += 1
            Iend -= 1

            # setup for maximum extent
            Imaxfwd = Istart
            Imaxback = Iend
            Vmaxfwd = 0.0
            Vmaxback = 0.0

            # determine ols in the forward direction
            fExy = 0.0
            fExx = 0.0
            fEx = 0.0
            fEy = 0.0

            distance = 0.0
            for i in range(Istart, Iend+1):
                Xc = float(i - Istart)
                Yc = cumr[i]
                fExy += Xc*Yc
                fExx += Xc*Xc
                fEx += Xc
                fEy += Yc

                if Xc > 0.0:
                    beta = (fExy - fEx*fEy/ (Xc+1.0)) / (fExx - fEx*fEx/ (Xc+1.0))
                    distance = dir * beta * Xc

                if distance > Vmaxfwd:
                    Vmaxfwd = distance
                    Imaxfwd = i


            # determine ols in the backward direction
            bExy = 0.0
            bExx = 0.0
            bEx = 0.0
            bEy = 0.0

            distance = 0.0
            for i in range(Iend, Istart-1, -1):
                Xc = float(Iend - i)
                Yc = cumr[i]
                bExy += Xc*Yc
                bExx += Xc*Xc
                bEx += Xc
                bEy += Yc

                if Xc > 0.0:
                    beta = (bExy - bEx*bEy/ (Xc+1.0)) / (bExx - bEx*bEx/ (Xc+1.0))
                    distance = -dir * beta * Xc

                if distance > Vmaxback:
                    Vmaxback = distance
                    Imaxback = i

            # if neither direction meets required minimum, zero out
            if Vmaxfwd < self.minamp and Vmaxback < self.minamp:
                 self._apply_label (labels, Istart, Iend, 0.0)
            else:
                # label forward region if meets size requirement
                if Vmaxfwd >= self.minamp:
                    self._apply_label (labels, Istart, Imaxfwd, dir)
                    self._apply_label (labels, Imaxfwd+1, Imaxback-1, 0.0)
                else:
                    self._apply_label (labels, Istart, Imaxback, 0.0)

                # label backward region if meets size requirement
                if Vmaxback >= self.minamp:
                    self._apply_label (labels, Imaxback, Iend, dir)
                else:
                    self._apply_label (labels, max(Imaxback, Imaxfwd+1), Iend, 0.0)

            Ipos = Iend+1


    cdef void _apply_label (self, double[:] labels, int Istart, int Iend, double dir):
        for i in range (Istart, Iend+1):
            labels[i] = dir

