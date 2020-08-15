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
from .GaussianHMM import GaussianHMM


class HMM2State(GaussianHMM):
    """
    A two state HMM
    Parameters
    ----------
    covar : :class:`float`
        The variance of each distribution (assumes all variances are the same)
    means : :class:`list`
        Means of the distributions
    ss_prob : :class:`float`
        Same state probability
    state_probs : :class:`ndarray`
        probability of each state
    Returns
    -------
    :class:`GuassianHMM`
        Ready to be used to predict
    """

    def __init__(
        self,
        covar = 0.25,
        means = [-0.5, 0.5],
        ss_prob=0.999,
        state_probs=np.array([1 / 2, 1 / 2])):

        means = np.array([[means[0], means[0]],[means[1], means[1]]])
        #means = np.array([[means[0]],[means[1]]])

        transition_matrix = np.array([[ss_prob, 1 - ss_prob], [1 - ss_prob, ss_prob]])

        cov_mat = covar * np.tile(np.eye(2), [2, 1, 1])
        #cov_mat = covar * np.tile(np.eye(1), [2, 1, 1])

        super().__init__(
            transition_matrix=transition_matrix,
            means=means,
            state_probs=state_probs,
            covar_matrix=cov_mat,
        )
