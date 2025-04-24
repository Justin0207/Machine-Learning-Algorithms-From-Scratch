# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 01:02:51 2024

@author: HP
"""
import numpy as np
class SVM:
    def __init__(self, lr = 0.001, beta = 0.01, n_iter = 1000):
        self.learning_rate = lr
        self.beta = beta
        self.n_iter = n_iter
        self.weights = None
        self.bias = None
        
    
        
    def fit(self, x, y):
        n_samples, n_features = x.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        for i in range(self.n_iter):
            for j, x_i in enumerate(x):
                y_new = np.where(y <= 0, -1, 1)
                y_pred = np.dot(x_i, self.weights) - self.bias
                if y_new[j] * y_pred >= 1:
                    dw = (2 * self.beta * self.weights)
                    db = 0
                    self.weights -= self.learning_rate * dw
                    self.bias -= self.learning_rate * db
                else:
                    dw = dw = (2 * self.beta * self.weights) - np.dot(y_new[j], x_i)
                    db = y_new[j]
                    self.weights -= self.learning_rate * dw
                    self.bias -= self.learning_rate * db
                
    def predict(self, X):
        approx = np.dot(X, self.weights) - self.bias
        y_pred = np.sign(approx)
        #y_pred = np.where(y == -1, 0, 1)
        return y_pred
    
def accuracy(y_pred, y_test):
    acc = np.sum(y_pred == y_test)/len(y_test)
    return acc
    
from sklearn import datasets
from sklearn.model_selection import train_test_split
X, y = datasets.load_breast_cancer(return_X_y = True)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 105)
lr = SVM()
lr.fit(x_train, y_train)
pred = lr.predict(x_test)
print(pred)
print(accuracy(pred, y_test))
    
