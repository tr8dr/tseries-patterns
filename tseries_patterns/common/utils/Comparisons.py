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

def isZero (v: float, eps=1e-7):
    return abs(v) < eps

def EQ (a: float, b: float, eps=1e-7):
    return abs(a-b) < eps

def GT (a: float, b: float, eps=1e-7):
    return (a-b) > eps

def GE (a: float, b: float, eps=1e-7):
    return (a-b) > -eps

def LT (a: float, b: float, eps=1e-7):
    return (a-b) < -eps

def LE (a: float, b: float, eps=1e-7):
    return (a-b) < eps

def vbetween (vec, a: float, b: float):
    return np.logical_and (vec >= a, vec <= b)


def vBetween (vec, a: float, b: float):
    return np.logical_and (vec >= a, vec <= b)

def vAnd (vecA, vecB):
    return np.logical_and (vecA, vecB)

def vOr (vecA, vecB):
    return np.logical_or (vecA, vecB)

def vLE (vec, v: float):
    return (vec <= v)

def vGE (vec, v: float):
    return (vec >= v)


def constrain(x, xmin, xmax):
    """
    Constrain a value x between [xmin,xmax]
    """
    if x < xmin:
        return xmin
    else:
        return min(x, xmax)


def frange(start: float, end: float, incr: float):
    """
    float range
    """
    eps = incr / 100
    while (end - start) > -eps:
        yield start
        start += incr


def rotate(l,n):
    """
    Rotate a list

    :param l: list to be rotated
    :param n: the rotation amount (+ shifts right and - shifts left)
    """
    return l[-n:] + l[:-n]


def ifelse (pred:bool, cons, alt):
    """
    If predicate consequent else alternative
    """
    if pred:
        return cons
    else:
        return alt


def OR (v, default):
    """
    return value unless is None
    """
    return v if v is not None else default
