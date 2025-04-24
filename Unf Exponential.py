# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 05:52:50 2024

@author: HP
"""

import numpy as np
import matplotlib.pyplot as plt
class Exponential:
    def __init__(self, lr=0.0000001, n_iters = 100000):
        self.lr = lr
        self.n_iters = n_iters
        self.bias = None
        self.weight = None
    def fit(self, x, y):
        n_samples, n_features = x.shape
        self.weight = np.zeros(n_features)
        self.bias = 0
        for i in range(self.n_iters):
            lin_pred = np.dot(x, self.weight)
            y_pred = np.exp(lin_pred) + self.bias
            dw = np.dot(x.T, (y_pred - y)) / n_samples
            db = np.sum(y_pred - y) / n_samples
            self.weight -= self.lr * dw
            self.bias -= self.lr * db
            
        return self
    def predict(self, x):
        lin_pred = np.dot(x, self.weight)
        y_pred = np.exp(lin_pred) + self.bias
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



reg = Exponential()
reg.fit(x_train, y_train)
exp_pred = reg.predict(x_test)
print('mean squared error of my model is', mse(exp_pred, y_test))
print(exp_pred)
print(len(x_test))

ri = Ridge()
ri.fit(x_train, y_train)
pred = ri.predict(x_test)
print('------------sklearn results------------')
print('mean squared error of sklearn model is', mse(pred, y_test))
print(pred)


plt.scatter(x_test[:, 4], exp_pred)
plt.show()