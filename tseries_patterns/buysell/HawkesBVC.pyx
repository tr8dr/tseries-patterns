#
# MIT License
#
# Copyright (c) 2020 Jonathan Shore
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
from scipy.stats import t as studentt

import plotnine
from plotnine import *

from ..common.rendering import scale_x_datetime_auto, new_grid
from ..common.utils import columnFor


cdef class HawkesBVC:
    """
    Use a hawkes process to model a self-exciting overlay on top of the BVC (bulk volume classifier)
    """
    cdef int _window
    cdef double _kappa
    cdef double _dof
    cdef object _metrics

    def __init__(self, window: int, kappa: float, dof = 0.25):
        """
        :param window lookback window for volatility calculation
        :param kappa decay factor (larger factor means for faster decay)
        :param dof degrees-of-freedom for student-t distribution (default 0.25)
        """
        self._window = window
        self._kappa = kappa
        self._dof = dof

    def eval (self, df: pd.DataFrame):
        """
        Compute BVC on bar / volume timeseries
        """

        if isinstance(df.index, pd.DatetimeIndex):
            times = df.index
        else:
            times = columnFor(df, ['stamp','time','Date','date','datetime'])

        prices = columnFor(df, ['close','Close','price'])
        cumr = np.log(prices / prices.iloc[0])
        r = cumr.diff().fillna(0.0)

        if "volume" in df.columns or "Volume" in df.columns:
            volume = columnFor(df, ['volume', 'Volume'])
        else:
            buyvol = columnFor(df, ['buyvolume', 'BuyVolume'])
            sellvol = columnFor(df, ['sellvolume', 'SellVolume'])
            volume = buyvol + sellvol

        sigma = r.rolling(self._window).std().fillna(0.0)
        alpha = np.exp (-self._kappa)
        labels = np.array([self._label(r[i], sigma[i]) for i in range(df.shape[0])])

        bvc = np.zeros(df.shape[0], dtype=float)

        self._compute_bvc (bvc, volume.values, labels, alpha)
        self._metrics = pd.DataFrame({'stamp': times, 'price': prices, 'bvc': bvc})
        return self._metrics

    def plot(
        self,
        color_price = 'darkgray',
        color_bvc = "#10a4f4",
        figsize = (10,8),
        title = ""):
        """
        Plot price and BSI.  Makes use of prior call to label function

        :param color_price: color for price series (dark gray default)
        :param colors_bvc: color for BVC
        :param figsize: size of graph
        :param title: title associated with graph
        :return:
        """
        df1 = pd.DataFrame({'stamp': self._metrics["stamp"], 'price': self._metrics["price"], 'pane': ' price'})
        df2 = pd.DataFrame({'stamp': self._metrics["stamp"], 'value': self._metrics["bvc"], 'pane': 'BVC'})

        plotnine.options.figure_size = figsize
        v = (ggplot() +
            geom_line(aes(x='stamp',y='price'), data=df1, color=color_price) +
            geom_line(aes(x='stamp',y='value'), data=df2, color=color_bvc) +
            scale_x_datetime_auto (df1["stamp"], figsize) +
            new_grid("pane ~ .", scales='free_y', height_ratios=[2,1]) +
            labs(title=title))

        return v


    def _label (self, r: float, sigma: float):
        if sigma > 0.0:
            cum = studentt.cdf(r / sigma, df = self._dof)
            return 2 * cum - 1.0
        else:
            return 0.0

    cdef _compute_bvc (self, double[:] out, double[:] volume, double[:] labels, double df):
        cdef double bvc = 0.0
        for i in range(volume.shape[0]):
            bvc = bvc * df + volume[i] * labels[i]
            out[i] = bvc



