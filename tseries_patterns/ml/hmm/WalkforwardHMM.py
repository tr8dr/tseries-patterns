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
import multiprocessing as mp
from math import ceil
from math import sin

from tseries_patterns.ml.hmm import HMM
from tseries_patterns.math.distributions import NormalDistribution


class WalkforwardHMM:
    def __init__(self, distributions, transition_matrix, state_probs, window=50):
        """
        Generic HMM is supplied with a distribution function, one per state

        :param distributions: list of distribution functions, each provided with observation vector yields log prob vector
        :param transition_matrix: transition probabilities
        :param state_probs: prior probability of being in any given state
        """
        self.distributions = distributions
        self.transition_matrix = transition_matrix
        self.state_probs = state_probs
        self.window = window


    def fit(self, x):
        """
        Train HMM, adjusting an variable not given
        :param X: series of states to train over
        """
        raise "fit not implemented"


    def predict(self, x, cores=12):
        """
        Predict state sequence
        """
        len = x.shape[0]
        chunk = max(1, int(ceil(float(len) / cores)))

        if isinstance(x, pd.Series):
            x = x.values

        def section(i):
            Iend = len - i
            Istart = max(0,Iend - chunk)
            return x[Istart:Iend]

        jobs = [
            [section(i), self.distributions, self.transition_matrix, self.state_probs, self.window]
            for i in range(0,len,self.window)]

        with mp.Pool() as pool:
            results = pool.map(_parallel_hmm, jobs)

        return pd.concat(results).reset_index(drop=True)


##
## External parallel function to run HMM on a core
##
def _parallel_hmm(job):
    section, dist, transitions, stateprob, window = job
    hmm = HMM (dist,transitions, stateprob)

    results = []
    for i in range(0,section.shape[0]):
        Iend = i+1
        Istart = max(Iend - window, 0)

        sub = section[Istart:Iend]
        states = hmm.predict(sub)
        results.append(states[-1])

    return pd.Series(results)


#
## test stuff
##

if __name__ == '__main__':
    data = []
    for i in range(0,10000):
        y = sin(i / 100.0 * 3.14152678)
        data.append(y)

    series = pd.Series(data)

    dist1 = NormalDistribution(-1.0, 0.7)
    dist2 = NormalDistribution(+1.0, 0.7)

    trans = np.array([
        [0.9999, 1 - 0.9999],
        [1 - 0.9999, 0.9999]])

    pi = [0.5, 0.5]

    hmm = WalkforwardHMM (distributions=[dist1.logf, dist2.logf], transition_matrix=trans, state_probs=pi, window = 50)
    labels = hmm.predict(series)

    print (labels.iloc[50])





