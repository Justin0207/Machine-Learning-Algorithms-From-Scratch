# -*- coding: utf-8 -*-
"""
Created on Tue May 14 16:42:47 2024

@author: HP
"""
import numpy as np
from scipy.stats import multivariate_normal
from pingouin import multivariate_normality
import pandas as pd
class GDA:
    def __init__(self):
        self.sigma = None
        self.mu = None
        self.phi = None
        
    def fit(self, x, y):
        n_samples, n_features = x.shape
        x = x.reshape(n_samples, -1)
        class_label = len(np.unique(y.reshape(-1)))
        self.mu = np.zeros((class_label, n_features))
        self.sigma = np.zeros((class_label, n_features, n_features))
        self.phi = np.zeros(class_label)
        for label in range(class_label):
            indices = (y == label)
            self.phi[label] = float(np.sum(indices)) / n_samples
            self.mu[label] = np.mean(x[indices, :], axis = 0)
            self.sigma[label] = np.cov(x[indices, :], rowvar = 0)
        return self.phi, self.mu, self.sigma
    
    def predict(self, x):
        x = x.reshape(x.shape[0], -1)
        class_label = self.mu.shape[0]
        scores = np.zeros((x.shape[0], class_label))
        
        for label in range(class_label):
            gaussian_prob = multivariate_normal(mean = self.mu[label], cov = self.sigma[label], allow_singular = True)
            for i, x_test in enumerate(x):
                scores[i, label] = np.log(self.phi[label]) + gaussian_prob.logpdf(x_test)
        predictions = np.argmax(scores, axis = 1)
        return predictions
def accuracy(y_pred, y_test):
    acc = np.sum(y_pred == y_test) / len(y_test)
    return acc

from sklearn import datasets
from sklearn.model_selection import train_test_split
X, y = datasets.load_iris(return_X_y = True)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 105)
a = X[:, 0]
b = X[:, 1]
c = X[:, 2]
d = X[:, 3]

df = pd.DataFrame([a, b, c, d])
# print(df)
print(multivariate_normality(df.T, alpha=.05))
lr = GDA()
lr.fit(x_train, y_train)
pred = lr.predict(x_test)
print(pred)
print(accuracy(pred, y_test))
        