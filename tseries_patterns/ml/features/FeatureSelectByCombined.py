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
from scipy.stats import wasserstein_distance


class FeatureSelectByCombined:

    def __init__(
        self,
        selectors: list,
        filter = lambda ith,corr: ith < 5 or corr < 0.9):
        """
        Select features by combining selectors

        :param selectors: list of selectors to evaluate
        :param filter: feature filter based on index and max correlation with prior features
        """
        self.selectors = selectors
        self.filter = filter


    def evaluate(self, df: pd.DataFrame, labels: pd.Series):
        """
        Determine most relevant features given feature set and labels

        :param df: feature data frame
        :param labels: class labels, one for each row
        """
        for selector in self.selectors:
            selector.evaluate (df, labels)

        selected_features = [self.selectors[0].selected.feature.iloc[0]]
        selected_importance = [self.selectors[0].selected.importance.iloc[0]]

        for selector in self.selectors:
            selected = selector.selected
            for fi in range(1, selected.shape[0]):
                feature = selected.feature.iloc[fi]
                importance = selected.importance.iloc[fi]
                ith = len(selected_features) + 1
                v = df[feature]
                maxcorr = df[[feature] + selected_features].corr().iloc[1:,0].max()

                if self.filter (ith, maxcorr):
                    selected_features.append (feature)
                    selected_importance.append (importance)

        self.selected = pd.DataFrame({
            "order": np.arange(len(selected_features)),
            "feature": selected_features,
            "importance": selected_importance }).reset_index(drop=True)

        return self.selected

