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

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from gcforest.gcforest import GCForest


class DeepRandomForestModel:

    def __init__(self, model: GCForest, scaler):
        self.model = model
        self.scaler = scaler

    def predict (self, xfeatures):
        if self.scaler is None:
            return self.model.predict(xfeatures)
        else:
            xfeatures_norm = self.scaler.transform(xfeatures)
            return self.model.predict(xfeatures_norm)


class DeepRandomForest:
    """
    Deep random forest as described by https://arxiv.org/abs/1702.08835v2
    """

    def __init__(
        self,
        nfolds = 5,
        nclasses = 2,
        nestimators = 10,
        maxlayers = 100,
        scaler = RobustScaler(quantile_range=(10, 90)),
        random_state = 0):
        """
        Create multi-layer deep random forest

        :param nfolds: number of folds in CV
        :param nclasses: number of classes
        :param nestimators: number of estimators
        :param maxlayers: maximum # of layers
        :param random_state:
        """
        self.scaler = scaler
        self.config = {
            'cascade': {
                'random_state': random_state,
                'max_layers': maxlayers,
                "early_stopping_rounds": 5,
                "n_classes": nclasses,
                "estimators": [
                    {"n_folds": nfolds, "type": "XGBClassifier", "n_estimators": nestimators, 'num_class': nclasses, "max_depth": 5,
                     "objective": "multi:softprob", "silent": True, "nthread": -1, "learning_rate": 0.1},
                    {"n_folds": nfolds, "type": "RandomForestClassifier", "n_estimators": nestimators, "max_depth": None,
                     "n_jobs": -1},
                    {"n_folds": nfolds, "type": "ExtraTreesClassifier", "n_estimators": nestimators, "max_depth": None, "n_jobs": -1},
                    {"n_folds": nfolds, "type": "LogisticRegression"}
                ]
            }
        }


    def fit (
        self,
        xtrain: pd.DataFrame,
        ytrain: pd.Series):
        """
        Fit model

        :param xtrain: training features
        :param ytrain: training labels
        """
        clf = GCForest(self.config)

        if self.scaler is None:
            clf.fit_transform(xtrain, ytrain)
        else:
            xtrain_norm = self.scaler.fit_transform(xtrain)
            clf.fit_transform(xtrain_norm, ytrain)

        return DeepRandomForestModel (clf, self.scaler)

