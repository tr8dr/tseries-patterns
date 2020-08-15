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


class HMM3State(GaussianHMM):
    """
       A three state HMM
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
        covar=0.25,
        means=[-0.7, 0.0, 0.7],
        ss_prob=0.999,
        state_probs=np.array([1 / 3, 1 / 3, 1 / 3])):

        # define the transition matrix based on the same-state probability
        one_jump = 2 / 3 * (1 - ss_prob)  # one jump, from outer to neutral
        two_jump = (1 - ss_prob) - one_jump  # from outer to other outer states
        zero_one = (1 - ss_prob) / 2.0  # from neutral to outer states

        means = np.array([[means[0]], [means[1]], [means[2]]])
        transition_matrix = np.array(
            [
                [ss_prob, one_jump, two_jump],
                [zero_one, ss_prob, zero_one],
                [two_jump, one_jump, ss_prob],
            ]
        )

        cov_mat = covar * np.tile(np.eye(1), [3, 1, 1])

        super().__init__(
            transition_matrix=transition_matrix,
            means=means,
            state_probs=state_probs,
            covar_matrix=cov_mat,
        )
