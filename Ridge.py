# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 11:15:10 2024

@author: HP
"""

import numpy as np
class RidgeRegression:
    def __init__(self, lr=1e-6, n_iters=1000, l2_penalty = 0.5):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        self.l2_penalty = l2_penalty
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for i in range(self.n_iters):
        
            y_pred = np.dot(X, self.weights) + self.bias
            
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y)) + (2*self.l2_penalty*self.weights)
            db = (1/n_samples) * np.sum(y_pred - y)
            
            self.weights = self.weights - (self.lr * dw)
            self.bias = self.bias - (self.lr * db)
            
    def predict(self, X):
        y_pred = np.dot(X, self.weights) + self.bias
        return y_pred
def mse(y_hat, y):
     mse = (y - y_hat)**2
     return np.mean(mse)
    
    
    
    
    
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
import pandas as pd

price = pd.read_csv(r'C:\Users\HP\Downloads\archive (2)\HousingData.csv')
price.dropna(inplace = True)
y = price['MEDV'].values
X = price.drop(['MEDV'], axis = 1).values

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 105)



reg = RidgeRegression()
reg.fit(x_train, y_train)
pred = reg.predict(x_test)
print('mean squared error of my model is', mse(pred, y_test))

ri = Ridge()
ri.fit(x_train, y_train)
pred = ri.predict(x_test)
print('------------sklearn results------------')
print('mean squared error of sklearn model is', mse(pred, y_test))

