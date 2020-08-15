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

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier


class RelabeledRandomForest:

    def __init__(self, ntrees=500, njobs=-1, maxdepth=20, nfolds = 5, sampling_function = None):
        self.ntrees = ntrees
        self.njobs = njobs
        self.maxdepth = maxdepth
        self.nfolds = nfolds
        self.sampling_function = sampling_function

    def fit (self, X: pd.DataFrame, y: pd.Series):
        w = X.shape[0]
        cuts = np.arange(0, w + 1, w / self.nfolds, dtype=int)
        cuts[-1] = w

        newlabels = np.zeros(w)
        for i in range(0, cuts.shape[0] - 1):
            xtesting = X.iloc[cuts[i]:cuts[i + 1]]
            ytesting = y.iloc[cuts[i]:cuts[i + 1]]

            xtraining = pd.concat([X.iloc[:cuts[i]], X.iloc[cuts[i + 1]:]])
            ytraining = pd.concat([y.iloc[:cuts[i]], y.iloc[cuts[i + 1]:]])

            if self.sampling_function is None:
                clf = RandomForestClassifier(
                    n_estimators=self.ntrees, max_depth=self.maxdepth,
                    random_state=1, n_jobs=-1, class_weight='balanced')
            else:
                clf = RandomForestClassifier(
                    n_estimators=self.ntrees, max_depth=self.maxdepth,
                    sampling_function = self.sampling_function,
                    random_state=1, n_jobs=-1, class_weight='balanced')

            model = clf.fit(xtraining, ytraining)
            ynew = clf.predict(xtesting)

            TP = (ynew == 1) & (ytesting == 1)
            FP = (ynew == 1) & (ytesting == 0)
            FN = (ynew == 0) & (ytesting == 1)
            TN = (ynew == 0) & (ytesting == 0)

            precision = TP.sum() / (TP.sum() + FP.sum()) * 100.0
            print("[%d/%d] precision: %1.1f%%" % (i + 1, cuts.shape[0] - 1, precision))

            ynew[FP] = 0.0
            newlabels[cuts[i]:cuts[i + 1]] = ynew

        self.clf = RandomForestClassifier(
            n_estimators=self.ntrees, max_depth=self.maxdepth,
            sampling_function=self.sampling_function,
            random_state=1, n_jobs=-1, class_weight='balanced')

        model = self.clf.fit (X, newlabels)
        self.y = newlabels
        self.X = X
        return model

    def importance(self, topk = None):
        scores = self.clf.feature_importances_
        names = self.X.columns

        indices = np.argsort(scores)[::-1]
        importance = pd.DataFrame({
            "feature": pd.Series(names)[indices],
            "importance": scores[indices]})

        if topk is None:
            return importance
        else:
            return importance.iloc[:topk]





