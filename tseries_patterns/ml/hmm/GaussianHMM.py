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
import hmmlearn



class GaussianHMM(hmmlearn.hmm.GaussianHMM):
    def __init__(self, transition_matrix = None, means = None, state_probs = None, covar_matrix = None):
        """
        In the GaussianHMM model, the process is model.fit(X) ==> model.predict(X_hat)
        However, a more useful implementation as it relates to signal smoothing
        is to define the HMM attributes like transition matrix, etc. and call
        model.predict(X_hat). This ensures stability as the model is "rolled"
        forward
        Parameters
        ----------
        transition_matrix : :class:`ndarray`
            (Optional) Transition matrix of probabilities, if not specified then must be trained
        means : :class:`ndarray`
            (Optional) 2D Means of the distributions, if not specified then must be trained
        state_probs : :class:`ndarray`
            (Optional) probability of each state, if not specified then must be trained
        covar_matrix : :class:`ndarray`
            (Optional) Covariance matrix with dimensions (n_components, n_dim, n_dim), if not specified then must be trained
        Returns
        -------
        :class:`GuassianHMM`
            Ready to be used to predict
        """
        n_components = len(state_probs)
        have = ""
        missing = ""
        if transition_matrix is None:
            missing = missing + "t"
        else:
            have = have + "t"
        if means is None:
            missing = missing + "m"
        else:
            have = have + "m"
        if covar_matrix is None:
            missing = missing + "c"
        else:
            have = have + "c"
        if state_probs is None:
            missing = missing + "s"
        else:
            have = have + "s"

        super().__init__(
            n_components=n_components,
            covariance_type="full",
            min_covar=1e-3,
            startprob_prior=0,
            transmat_prior=0,
            means_prior=0,
            covars_prior=0.0,
            algorithm="viterbi",
            params=missing,
            init_params=have
        )

        self.transmat_ = transition_matrix
        self.means_ = means
        self.startprob_ = state_probs
        self.covars_ = covar_matrix

    def fit(self, X):
        """
        Train HMM, adjusting an variable not given
        :param X: series of states to train over
        """
        super().fit(X)

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

