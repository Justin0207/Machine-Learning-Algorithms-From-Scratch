# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 04:17:36 2024

@author: HP
"""
import numpy as np

class MultiClassClassification:
    def __init__(self, lr=0.001, n_iters=10000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        
        
    def fit(self, x, y):
        n_samples, n_features = x.shape
        self.y_unique = list(np.unique(y))
        self.weights = np.zeros((n_features, len(self.y_unique)))
        self.bias = 0
        y = self.one_hot_encoding(y)
        for epoch in range(self.n_iters):
            lin_pred = np.dot(x, self.weights) + self.bias
            y_pred = self.softmax(lin_pred)
            
            dw = (1/n_samples) * np.dot(x.T, (y_pred - y))
            db = (1/n_samples) * np.sum((y_pred - y))
            
            self.weights = self.weights - self.lr * dw
            self.bias = self.bias - self.lr * db
            
    
    def one_hot_encoding(self, y):
        encoded = np.zeros((len(y), len(self.y_unique)))
        for i, c in enumerate(y):
            encoded[i][self.y_unique.index(c)] = 1
        return encoded
    def softmax(self, z):
        result = np.exp(z) / np.sum(np.exp(z), axis = 1).reshape(-1, 1)
        return result
    
    def predict(self, x):
        lin_pred = np.dot(x, self.weights) + self.bias
        y_pred = self.softmax(lin_pred)
        max_index = [np.argmax(i) for i in y_pred]
        labels = [self.y_unique[i] for i in max_index]
        return labels
    
def accuracy(y_pred, y_test):
    acc = np.sum(y_pred == y_test)/len(y_test)
    return acc
    
from sklearn import datasets
from sklearn.model_selection import train_test_split
X, y = datasets.load_iris(return_X_y = True)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 105)
lr = Multi()
lr.fit(x_train, y_train)
pred = lr.predict(x_test)
print(pred)
print(accuracy(pred, y_test))
            
        