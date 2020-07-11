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

import plotnine
import numpy as np
import pandas as pd

def scale_x_datetime_auto(times: pd.Series, figsize=(12,10)):
    """
    Automatically set breaks and format based on duration of series
    """
    width = figsize[0]

    dt = (times.iloc[-1] - times.iloc[0]).total_seconds()
    mins = dt / 60
    hours = dt / 3600
    days = dt / 3600 / 24

    if days > 10:
        fmt = "%Y-%m-%d"
        breaks = "%1.0f days" % max(np.round(days / (width / 2.0)), 1.0)
        return plotnine.scale_x_datetime(date_labels=fmt, date_breaks=breaks)
    elif days > 1.0:
        fmt = "%Y-%m-%d %H:%M"
        breaks = "%1.0f hours" % max(np.round(hours / (width / 2.0)), 1.0)
        return plotnine.scale_x_datetime(date_labels=fmt, date_breaks=breaks)
    elif hours > 1:
        fmt = "%H:%M"
        breaks = "%1.0f minutes" % max(np.round(mins / (width / 1.5)), 1.0)
        return plotnine.scale_x_datetime(date_labels=fmt, date_breaks=breaks)
    else:
        fmt = "%H:%M:%S"
        breaks = "%1.0f minutes" % max(np.round(mins / (width / 1.5)), 1.0)
        return plotnine.scale_x_datetime(date_labels=fmt, date_breaks=breaks)
