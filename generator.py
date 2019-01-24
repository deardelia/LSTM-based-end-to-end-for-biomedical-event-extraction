# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 16:02:56 2017

@author: Administrator
"""
import numpy as np
from load_data import load_embedding, load_voc, load_train_data
from sklearn.model_selection import KFold
import configurations as config


# ** 3.build the data generator
class BatchGenerator(object):
    """ Construct a Data generator. The input X, y should be ndarray or list like type.
    
    Example:
        Data_train = BatchGenerator(X=X_train_all, y=y_train_all, shuffle=False)
        Data_test = BatchGenerator(X=X_test_all, y=y_test_all, shuffle=False)
        X = Data_train.X
        y = Data_train.y
        or:
        X_batch, y_batch = Data_train.next_batch(batch_size)
     """ 
    
    def __init__(self, X,X_tag, y, shuffle=False):
        if type(X) != np.ndarray:
            X = np.asarray(X)
        if type(X_tag) != np.ndarray:
            X_tag = np.asarray(X_tag)
        if type(y) != np.ndarray:
            y = np.asarray(y)
        self._X = X
        self._X_tag = X_tag
        self._y = y
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._number_examples = self._X.shape[0]
        self._shuffle = shuffle
        if self._shuffle:
            new_index = np.random.permutation(self._number_examples)
            self._X = self._X[new_index]
            self._X_tag = self._X_tag[new_index]
            self._y = self._y[new_index]
                
    @property
    def X(self):
        return self._X
    @property
    def X_tag(self):
        return self._X_tag
    
    @property
    def y(self):
        return self._y
    
    @property
    def num_examples(self):
        return self._number_examples
    
    @property
    def epochs_completed(self):
        return self._epochs_completed
    
    def next_batch(self, batch_size):
        """ Return the next 'batch_size' examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._number_examples:
            # finished epoch
            self._epochs_completed += 1
            # Shuffle the data 
            if self._shuffle:
                new_index = np.random.permutation(self._number_examples)
                self._X = self._X[new_index]
                self._X_tag = self._X_tag[new_index]
                self._y = self._y[new_index]
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._number_examples
        end = self._index_in_epoch
        return self._X[start:end],self._X_tag[start:end], self._y[start:end]

