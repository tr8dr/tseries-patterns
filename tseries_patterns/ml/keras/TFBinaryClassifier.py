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
import tensorflow as tf
import keras.backend as K

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
from sklearn.preprocessing import StandardScaler, RobustScaler
from.PerformanceMeasures import precision, recall, f1score, fbeta
from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold


class TFBinaryClassifier:
    """
    multi-layer RELU based ANN with dropout between each layer
    """

    def __init__(
        self,
        nfeatures: int,
        layers: list,
        hidden='relu',
        output='sigmoid',
        scaler = RobustScaler(quantile_range=(10, 90)),
        dropout=0.2,
        optimizer = tf.keras.optimizers.Adam(decay=1e-6),
        metrics=[fbeta,precision,recall]):
        """
        Create multi-layer RELU based ANN with dropout between each layer

        :param nfeatures: dimension of input features
        :param layers: list of layer sizes
        :param hidden: activation type for inner layers
        :param output: activation type for output layer
        :param dropout: dropout rate
        :param metrics: metrics used to score
        :return: model
        """
        self.scaler = scaler
        self.model = Sequential()
        self.model.add (Dense (layers[0], input_dim = nfeatures, activation=hidden))
        self.model.add (Dropout(rate = dropout))
        self.model.add (BatchNormalization())

        for dim in layers[1:]:
            self.model.add (Dense (dim, activation = hidden))
            self.model.add (Dropout(rate = dropout))
            self.model.add (BatchNormalization())

        self.model.add (Dense(1, activation=output))
        self.model.compile (
            loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=metrics)


    def fit (
        self,
        xtrain: pd.DataFrame,
        ytrain: pd.Series,
        batchsize = 500,
        epochs = 200,
        class_weight = { 0: 1, 1: 1 },
        xvalidate = None, yvalidate = None):
        """
        Fit model

        :param xtrain: training features
        :param ytrain: training labels
        :param batchsize: size of batches for each epoch
        :param epochs: number of iterations
        :param xvalidate: validation features (optional)
        :param yvalidate: validation labels (optional)
        """

        self.xtrain = xtrain
        self.ytrain = ytrain
        self.epochs = epochs
        self.batchsize = batchsize

        xtrain_norm = self.scaler.fit_transform(xtrain)
        if xvalidate is None:
            self.model.fit(
                xtrain_norm, ytrain.values,
                class_weight=class_weight,
                batch_size=batchsize, epochs=epochs)
        else:
            xvalidate_norm = self.scaler.transform(xvalidate)
            self.model.fit(
                xtrain_norm, ytrain.values,
                batch_size=batchsize, epochs=epochs,
                class_weight=class_weight,
                validation_data=(xvalidate_norm, yvalidate.values))


    def predict (self, xfeatures: pd.DataFrame, threshold = None):
        """
        Predict 0 or 1 labels (if threshold provided), otherwise likelihood

        :param xfeatures: feature set
        :param threshold: threshold above which label = 1, otherwise 0
        :return:
        """
        xfeatures_norm = self.scaler.transform(xfeatures)
        p = self.model.predict (xfeatures_norm).flatten()
        if threshold is None:
            return pd.Series(p)
        else:
            return (pd.Series(p) >= threshold) * 1.0


    def kfoldCV (
        self,
        xtrain: pd.DataFrame,
        ytrain: pd.Series,
        batchsize=500,
        epochs=200,
        nfolds = 10):
        """
        Do K-fold cross-validation model

        :param xtrain: training features
        :param ytrain: training labels
        :param batchsize: size of batches for each epoch
        :param epochs: number of iterations
        :param nfolds:
        :return:
        """
        self.xtrain = xtrain
        self.ytrain = ytrain
        self.epochs = epochs
        self.batchsize = batchsize

        xtrain_norm = self.scaler.fit_transform(xtrain)
        estimator = KerasClassifier (build_fn = self.model, epochs = epochs, batch_size = batchsize)
        kfold = KFold (nfolds)
        results = cross_val_score(estimator, xtrain_norm, ytrain, cv=kfold)

        return results