# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 08:42:23 2024

@author: HP
"""

import numpy as np
from collections import Counter
def euclidean_distance(x1, x2):
    distance = np.sqrt(np.sum((x2 - x1) **2))
    return distance
class KNN:
    def __init__(self, k=3):
        self.k = k
    def fit(self, x, y):
        self.X_train = x
        self.y_train = y
    def predict(self, X):
        predictions = [self._predict(j) for j in X]
        return predictions
    def _predict(self, x):
        #compute the distances
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        # sort the indices
        k_indices = np.argsort(distances)[: self.k]
        # Get their labels
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = Counter(k_nearest_labels).most_common()
        return most_common[0][0]
        
        
        