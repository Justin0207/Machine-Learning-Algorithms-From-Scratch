# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 04:40:36 2024

@author: HP
"""
import numpy as np
from collections import Counter
def euclidean_distance(x1, x2):
    dist = np.sqrt(np.sum((x1 - x2)**2))
    return dist 
def accuracy(y_test, y_pred):
    acc = np.sum(y_pred == y_test) / len(y_test)
    return acc
    
class KNN:
    def __init__(self, k=5):
        self.k = k
        
    def fit(self, x, y):
        self.X_train = x
        self.y_train = y
        
    def predict(self, X):
        prediction = [self.compute(x) for x in X]
        return prediction
    
    def compute(self, x):
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        neighbors_indices = np.argsort(distances)[: self.k]
        neighbors = [self.y_train[i] for i in neighbors_indices]
        neighbors_average = np.mean(neighbors)
        return neighbors_average
    
def mse(y_hat, y):
     mse = (y - y_hat)**2
     return np.mean(mse)
    
    
    

        
    