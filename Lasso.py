# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 10:27:56 2024

@author: HP
"""

import numpy as np
class LassoRegression:
    def __init__(self, lr=0.000001, n_iters=1000, l1_penalty=10):
        self.lr = lr
        self.n_iters = n_iters
        self.bias = None
        self.weight = None
        self.l1_penalty = l1_penalty
    def fit(self, x, y):
        n_samples, n_features = x.shape
        self.weight = np.zeros(n_features)
        self.bias = 0
        for i in range(self.n_iters):
            y_pred = np.dot(x, self.weight) + self.bias
            dw = np.zeros(n_features)
            for j in range(n_features):
                if dw[j] >= 0:
                    dw = (1/n_samples) * np.dot(x.T, y_pred - y) + self.l1_penalty
                else:
                    dw = (1/n_samples) * np.dot(x.T, y_pred - y) - self.l1_penalty
            db = (1/n_samples) * np.sum(y_pred - y)
            self.weight = self.weight - (self.lr * dw)
            self.bias = self.bias - (self.lr * db)
        return self
            
    def predict(self, x):
        y_pred = np.dot(x, self.weight) + self.bias
        return y_pred

def mse(y_hat, y):
     mse = (y - y_hat)**2
     return np.mean(mse)
    
    
    
    
    
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.linear_model import Lasso

price = pd.read_csv(r'C:\Users\HP\Downloads\archive (2)\HousingData.csv')
price.dropna(inplace = True)
y = price['MEDV'].values
X = price.drop(['MEDV'], axis = 1).values

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 105)



reg = LassoRegression()
reg.fit(x_train, y_train)
pred = reg.predict(x_test)
print(mse(pred, y_test))
print('mean squared error of my model is', mse(pred, y_test))



lass = Lasso()
lass.fit(x_train, y_train)
pred = lass.predict(x_test)
print('------------sklearn results------------')
print('mean squared error of sklearn model is', mse(pred, y_test))