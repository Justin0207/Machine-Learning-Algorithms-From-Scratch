# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 21:39:20 2024

@author: HP
"""
import numpy as np



class Perceptron:
    def __init__(self, lr=0.00001, n_iters=10000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        
    def activation_func(self, x):
        step = np.where(x>0, 1, 0)
        return step
        
    def fit(self, x, y):
        n_samples, n_features = x.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        y_ = self.activation_func(y)
        for i in range(self.n_iters):
            for j in range(n_samples):
                lin_pred = np.dot(x[j], self.weights) + self.bias
                y_pred = self.activation_func(lin_pred)
                
                self.weights = self.weights + ((self.lr * (y_[j] - y_pred)) * x[j])
                self.bias = self.bias + (self.lr * (y_[j] - y_pred))
    def predict(self, x):
        lin_pred = np.dot(x, self.weights) + self.bias
        y_pred = self.activation_func(lin_pred)
        return y_pred
    
def accuracy(y_pred, y_test):
    acc = np.sum(y_pred == y_test)/len(y_test)
    return acc
    
from sklearn import datasets
from sklearn.model_selection import train_test_split
X, y = datasets.load_breast_cancer(return_X_y = True)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 105)
lr = Perceptron()
lr.fit(x_train, y_train)
pred = lr.predict(x_test)
print(pred)
print(accuracy(pred, y_test))
                
                
        