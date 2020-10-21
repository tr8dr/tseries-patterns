#
# MIT License
#
# Copyright (c) 2011 Jonathan Shore
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

from ..common.rendering import scale_x_datetime_auto, new_grid
from ..common.utils import columnFor


cdef class HawkesBSI:
    """
    Use a hawkes process to model a self-exciting buy/sell imbalance signal
    """
    cdef double _kappa
    cdef object _metrics

    def __init__(self, kappa):
        """
        :param kappa decay factor (larger factor means for faster decay)
        """
        self._kappa = kappa

    def eval (self, df: pd.DataFrame):
        """
        Compute buy/sell imbalance on bar / volume timeseries
        """

        if isinstance(df.index, pd.DatetimeIndex):
            times = df.index
        else:
            times = columnFor(df, ['stamp','time','Date','date','datetime'])

        prices = columnFor(df, ['close','Close','price'])
        buyvol = columnFor(df, ['buyvolume', 'BuyVolume'])
        sellvol = columnFor(df, ['sellvolume', 'SellVolume'])

        dv = buyvol - sellvol
        alpha = np.exp (-self._kappa)
        bsi = np.zeros(dv.shape[0], dtype=float)

        self._compute_bsi (bsi, np.array(dv.values, dtype=float), alpha)
        self._metrics = pd.DataFrame({'stamp': times, 'price': prices, 'bsi': bsi})
        return self._metrics

    def plot(
        self,
        Tstart = None,
        Tend = None,
        color_price = 'darkgray',
        color_bsi = "#10a4f4",
        figsize = (10,8),
        title = ""):
        """
        Plot price and BSI.  Makes use of prior call to label function

        :param color_price: color for price series (dark gray default)
        :param colors_bsi: color for BSI
        :param figsize: size of graph
        :param title: title associated with graph
        :return:
        """
        if Tstart is None:
            Tstart = self._metrics["stamp"].iloc[0]
        if Tend is None:
            Tend = self._metrics["stamp"].iloc[-1]

        sub = self._metrics.loc[(self._metrics.stamp >= Tstart) & (self._metrics.stamp <= Tend)]

        df1 = pd.DataFrame({'stamp': sub["stamp"], 'price': sub["price"], 'pane': ' price'})
        df2 = pd.DataFrame({'stamp': sub["stamp"], 'value': sub["bsi"], 'pane': 'BSI'})

        plotnine.options.figure_size = figsize
        v = (ggplot() +
            geom_line(aes(x='stamp',y='price'), data=df1, color=color_price) +
            geom_line(aes(x='stamp',y='value'), data=df2, color=color_bsi) +
            scale_x_datetime_auto (df1["stamp"], figsize) +
            new_grid("pane ~ .", scales='free_y', height_ratios=[2,1]) +
            labs(title=title))

        return v


    cdef _compute_bsi (self, double[:] out, double[:] dv, double df):
        cdef double bsi = 0.0
        for i in range(dv.shape[0]):
            bsi = bsi * df + dv[i]
            out[i] = bsi



