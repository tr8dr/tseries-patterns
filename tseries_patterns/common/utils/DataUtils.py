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

import numpy as np
import pandas as pd
from collections import Iterable
from scipy.stats import *

def columnFor (df: pd.DataFrame, names: list):
    """
    Find named column from a list of alternatives
    """
    for id in names:
        if id in df.columns:
            return df[id]
    raise Exception (f"could not find {names[0]} column in supplied dataframe")

def ncols(series):
    """
    Determine # of columns
    """
    if not hasattr(series, 'shape'):
        return 1
    if len(series.shape) == 1:
        return 1
    else:
        return series.shape[1]

def nrows(series):
    """
    Determine # of rows
    """
    if not hasattr(series, 'shape'):
        return len(series)
    else:
        return series.shape[0]


def breaks(series, mingap = 12*3600):
    """
    Get the row indices where there are at least k-second gaps

    :param series: series to determine breaks on
    :return: list of breaks
    """
    dt = np.diff(series.index.astype(np.int64)/1e9)

    isbreak = np.concatenate ((np.array([True]), dt[:-1] >= mingap, np.array([True])))
    indices, = np.where (isbreak)
    return indices


def toColumnVector(vec) -> np.array:
    """
    Convert vector to column vector if currently a row vector
    """
    if not hasattr(vec, 'shape'):
        vec = np.array(vec)

    shape = vec.shape
    if len(shape) == 1:
        return np.reshape (vec, (shape[0], 1))
    elif shape[0] == 1:
        return np.transpose(vec)
    else:
        return vec


def toRowVector(vec) -> np.array:
    """
    Convert vector to row vector if currently a column vector
    """
    if isinstance(vec, pd.Series):
        return vec.values
    elif not hasattr(vec, 'shape'):
        return np.array(vec) if isinstance(vec, Iterable) else np.array([vec])
    else:
        return vec.flatten()


def cbind(*serieslist):
    """
    compose a list of column vectors into a matrix
    :param serieslist:
    :return:
    """
    vectors = [toColumnVector(vec) for vec in serieslist if vec is not None]
    return np.hstack(vectors)


def c(*serieslist):
    """
    concatenate a list of vectors
    """
    vectors = [toRowVector(vec) for vec in serieslist]
    return np.concatenate(vectors)


def summary(series):
    """
    Descriptive statistics for series
    """
    idx = ['mean', 'std', 'skew', 'kurtosis', 'min', '25%', 'median', '75%', 'max']

    def statistics (v):
        v = v[~np.isnan(v)]
        return np.array([
            np.mean(v), np.std(v), skew(v), kurtosis(v),
            np.min(v),
            np.percentile(v, 25), np.percentile(v, 50), np.percentile(v, 75),
            np.max(v)])

    if len(series.shape) == 1 or series.shape[1] == 1:
        return pd.DataFrame(statistics(series), index=idx, columns=None)

    data = None
    columns = series.columns
    for ci in columns:
        v = series[[ci]].values
        stats = statistics(v)
        snew = pd.DataFrame(stats, index=idx, columns=[ci])
        if data is None:
            data = snew
        else:
            data = pd.merge (data, snew, left_index=True, right_index=True, how='outer')

    return data




