# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 20:28:29 2024

@author: HP
"""

#################SGD With Momentum #########################
# import numpy as np
# class LinearRegression:
#     def __init__(self, lr=0.0000001, n_iters=100000, epochs = 1000):
#         self.lr = lr
#         self.n_iters = n_iters
#         self.weights = None
#         self.bias = None
#         self.weight = []
#         self.rho = None
#     def fit(self, X, y):
#         n_samples, n_features = X.shape
#         self.weights = np.zeros(n_features)
#         self.bias = 0
#         self.rho = 0.9999
#         vx = 0
        
        
#         for i in range(self.n_iters):
#             y_pred = np.dot(X, self.weights) + self.bias

#             dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
#             vx = self.rho * vx + dw
#             db = (1/n_samples) * np.sum(y_pred - y)
            
#             self.weights = self.weights - (self.lr * vx)
#             self.bias = self.bias - (self.lr * db)
#             self.weight.append(self.weights)
                
                        
            
#     def predict(self, X):
#         y_pred = np.dot(X, self.weights) + self.bias
#         return y_pred
# def mse(y_hat, y):
#      mse = (y - y_hat)**2
#      return np.mean(mse)
    
######################### SGD ###############################    
import numpy as np
class LinearRegression:
    def __init__(self, lr=0.0000001, n_iters=100000, epochs = 1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        self.weight = []
        self.rho = None
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        
        for i in range(self.n_iters):
            y_pred = np.dot(X, self.weights) + self.bias

            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)
            
            self.weights = self.weights - (self.lr * dw)
            self.bias = self.bias - (self.lr * db)
            self.weight.append(self.weights)
                
                        
            
    def predict(self, X):
        y_pred = np.dot(X, self.weights) + self.bias
        return y_pred
def mse(y_hat, y):
     mse = (y - y_hat)**2
     return np.mean(mse)    
    
    
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.linear_model import LinearRegression as linreg

price = pd.read_csv(r'C:\Users\HP\Downloads\archive (2)\HousingData.csv')
price.dropna(inplace = True)
y = price['MEDV'].values
X = price.drop(['MEDV'], axis = 1).values

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 105)



reg = LinearRegression()
reg.fit(x_train, y_train)
pred = reg.predict(x_test)
print('mean squared error of my model is', mse(pred, y_test))














