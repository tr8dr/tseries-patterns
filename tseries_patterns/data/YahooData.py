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

from math import log
from datetime import datetime
from pandas_datareader import data as pdr
import yfinance


class YahooData:
    """
    Load Yahoo Finance Stock Data
    """

    @staticmethod
    def getOHLC(stock: str, Tstart=datetime(1999, 1, 1), Tend=datetime.now()):
        """
        Get adjusted OHLC bars for stock and date range
        """
        df = pdr.get_data_yahoo([stock], start=Tstart, end=Tend)
        df.columns = ["adjclose", "close", "high", "low", "open", "volume"]

        open = df.open
        high = df.high
        low = df.low
        close = df.close
        adjclose = df.adjclose
        volume = df.volume

        # create adjusted OHLCV
        ratio = adjclose / close
        newdf = pd.DataFrame({
            'open': open * ratio,
            'high': high * ratio,
            'low': low * ratio,
            'close': close * ratio,
            'volume': volume * ratio}, index=df.index)

        return newdf


    @staticmethod
    def getPrices (stocklist: list, Tstart = datetime(1999,1,1), Tend = datetime.now()):
        """
        Get portfolio of prices
        """
        df = None
        for stock in stocklist:
            bars = YahooData.getOHLC(stock, Tstart, Tend)

            values = bars["adjclose"].values
            index = bars.index
            part = pd.DataFrame (data=values, index=index, columns=[stock])

            if df is None:
                df = part
            else:
                df = df.merge(part, left_index=True, right_index=True)

        return df


    @staticmethod
    def getReturns (stocklist: list, type = "cumr", Tstart = datetime(1999,1,1), Tend = datetime.now()):
        """
        Get portfolio of returns
        """
        df = None
        for stock in stocklist:
            bars = YahooData.getOHLC(stock, Tstart, Tend)

            x = bars["adjclose"].values
            if type == "cumr" or type == "cum":
                r = [0.0] + [log(x[i] / x[0]) * 1e4 for i in range(1,x.size)]
            elif type == "r" or type == "return":
                r = [0.0] + [log(x[i] / x[i-1]) * 1e4 for i in range(1,x.size)]
            else:
                raise ("unknown return type: %s" % type)

            index = bars.index
            part = pd.DataFrame (data=r, index=index, columns=[stock])

            if df is None:
                df = part
            else:
                df = df.merge(part, left_index=True, right_index=True)

        return df
