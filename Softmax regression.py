# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 20:47:07 2024

@author: HP
"""
import numpy as np
class MultiNomial:
    def __init__(self, lr=0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.bias = None
        self.weights = None
        
    def softmax(self, z):
        probabilities = np.exp(z) / np.sum(np.exp(z), axis = 1).reshape(-1, 1)
        return probabilities
    
    def one_hot(self, y):
        y_unique_val = list(np.unique(y))
        encoded = np.zeros((len(y), len(y_unique_val)))
        for i, c in enumerate(y):
            encoded[i][y_unique_val.index(c)] = 1
        return encoded
    
    def fit(self, x, y):
        n_samples, n_features = x.shape
        self.y_unique = np.unique(y)
        self.weights = np.zeros((n_features, len(self.y_unique)))
        self.bias = 0
        y = self.one_hot(y)
        for i in range(self.n_iters):
            lin_pred = np.dot(x, self.weights) + self.bias
            y_pred = self.softmax(lin_pred)
            
            dw = (1/n_samples) * np.dot(x.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)
            
            self.weights = self.weights - (self.lr * dw)
            self.bias = self.bias - (self.lr * db)
            
        return self
    def predict(self, x):
        lin_pred = np.dot(x, self.weights) + self.bias
        y_pred = self.softmax(lin_pred)
        res = [np.argmax([i]) for i in y_pred]
        classes = [self.y_unique[j] for j in res]
        return classes
    
def accuracy(y_pred, y_test):
    acc = np.sum(y_pred == y_test)/len(y_test)
    return acc
            
            
            
            
from sklearn import datasets
from sklearn.model_selection import train_test_split
X, y = datasets.load_iris(return_X_y = True)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 105)
lr = MultiNomial()
lr.fit(x_train, y_train)
pred = lr.predict(x_test)
print(pred)
print(accuracy(pred, y_test))

            