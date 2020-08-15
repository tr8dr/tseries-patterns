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
import hmmlearn.hmm

from hmmlearn.hmm import _BaseHMM



class HMM(_BaseHMM):
    def __init__(self, distributions, transition_matrix, state_probs):
        """
        Generic HMM is supplied with a distribution function, one per state

        :param distributions: list of distribution functions, each provided with observation vector yields log prob vector
        :param transition_matrix: transition probabilities
        :param state_probs: prior probability of being in any given state
        """
        self.nstates = len(state_probs)

        super().__init__(
            n_components=self.nstates,
            startprob_prior=0,
            transmat_prior=0)

        self.distributions = distributions
        self.transmat_ = transition_matrix
        self.startprob_ = state_probs


    def fit(self, X):
        """
        Train HMM, adjusting an variable not given
        :param X: series of states to train over
        """
        raise "fit not implemented"


    def predict(self, srs):
        """
        The hmm_model.predict() functionality has the common issue where
        Series need to be be converted to values & reshaped, this does that
        conversion and then uses the ``super().predict`` method from the
        hmm module
        Parameters
        ----------
        srs : :class:`Series` or :class:`ndarray`
            to predict
        Returns
        -------
        :class:`Series` or :class:`ndarray`
            Prediction
        """
        if isinstance(srs, pd.Series):
            arr = srs.values.reshape(-1, 1)
            res = pd.Series(super().predict(arr), index=srs.index)

        elif isinstance(srs, np.ndarray):
            arr = srs.reshape(-1, 1)
            res = super().predict(arr)

        return res


    def _compute_log_likelihood(self, X):
        n = X.shape[0]
        log_prob = np.empty((n, self.nstates))

        for ci in range(0, self.nstates):
            log_prob[:,ci] = self.distributions[ci] (X).flatten()

        return log_prob

