# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 04:32:23 2024

@author: HP
"""
import numpy as np
class LogisticRegression:
    def __init__(self, lr=0.0001, n_iters=10000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        
    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))
    
    def fit(self, x, y):
        n_samples, n_features = x.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        
        for i in range(self.n_iters):
            lin_pred = np.dot(x, self.weights) + self.bias
            predictions = self.sigmoid(lin_pred)
            
            dw = (np.dot(x.T, predictions - y)) / n_samples
            db = (np.sum(predictions - y)) / n_samples
            self.weights = self.weights - self.lr * dw
            self.bias = self.bias - self.lr * db
            
    def predict(self, x):
        linear_pred = np.dot(x, self.weights) + self.bias
        y_pred = self.sigmoid(linear_pred)
        class_pred = [0 if y < 0.5 else 1 for y in y_pred]
        return class_pred
    
def accuracy(y_pred, y_test):
    acc = np.sum(y_pred == y_test)/len(y_test)
    return acc



