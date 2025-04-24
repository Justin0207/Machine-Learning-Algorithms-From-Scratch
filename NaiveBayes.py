# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 21:19:54 2024

@author: HP
"""

import numpy as np
class NaiveBayes:
    def fit(self, x, y):
        n_samples, n_features = x.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)
        self._mean = np.zeros((n_classes, n_features), dtype= np.float64)
        self._var = np.zeros((n_classes, n_features), dtype= np.float64)
        self._priors = np.zeros(n_classes, dtype= np.float64)
        for index, c in enumerate(self._classes):
            x_c = x[y == c]
            self._mean[index, :] = x_c.mean(axis = 0)
            self._var[index, :] = x_c.var(axis = 0)
            self._priors[index] = x_c.shape[0] / float(n_samples)
            
            
    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)
            
        
    def _predict(self, x):
        posteriors = [ ]
        for index in range(len(self._classes)): 
            prior = np.log(self._priors[index])
            posterior = np.sum(np.log(self._pdf(index, x)))
            posterior = posterior + prior
            posteriors.append(posterior)
            
        return self._classes[np.argmax(posteriors)]
    
    
    def _pdf(self, class_idx, x):
        mean = self._mean[class_idx]
        var = self._var[class_idx]
        numerator = np.exp(-((x - mean) **2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator
    
    
def accuracy(y_pred, y_test):
    acc = np.sum(y_pred == y_test)/len(y_test)
    return acc
    
    
    
    
    
    
    
from sklearn import datasets
from sklearn.model_selection import train_test_split
X, y = datasets.load_iris(return_X_y = True)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 105)
lr = NaiveBayes()
lr.fit(x_train, y_train)
pred = lr.predict(x_test)
print(pred)
print(accuracy(pred, y_test))
            
            
            
    