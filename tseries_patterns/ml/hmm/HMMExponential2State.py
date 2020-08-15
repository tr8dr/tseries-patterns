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
from .HMM import HMM

from research.math.distributions import ExponentialDistribution


class HMMExponential2State(HMM):
    """
    A two state HMM using exponential distributions
    """

    def __init__(
        self,
        decay = 2.0,
        ss_prob=0.9999,
        state_probs=np.array([1 / 2, 1 / 2])):

        transition_matrix = np.array([[ss_prob, 1 - ss_prob], [1 - ss_prob, ss_prob]])
        short = ExponentialDistribution(-1.0, decay, 1.0)
        long = ExponentialDistribution(+1.0, decay, -1.0)

        super().__init__(
            distributions = [short.logf, long.logf],
            transition_matrix=transition_matrix,
            state_probs=state_probs)
