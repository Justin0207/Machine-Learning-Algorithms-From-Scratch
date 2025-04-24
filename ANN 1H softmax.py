# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 20:14:15 2024

@author: Favour
"""

import numpy as np
import matplotlib.pyplot as plt

class DeepNeuralNetwork:
    def __init__(self, sizes, activation='relu', optimizer='momentum', epochs=1000, batch_size=32, lr=0.01, beta=0.9):
        self.sizes = sizes
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.beta = beta
        self.optimizer = optimizer.lower()
        self.losses = []

        if self.optimizer == 'momentum':
            self.momentum_opt = {
                'W1': np.zeros((sizes[1], sizes[0])),
                'b1': np.zeros((sizes[1], 1)),
                'W2': np.zeros((sizes[2], sizes[1])),
                'b2': np.zeros((sizes[2], 1))
            }
        
        if activation == 'relu':
            self.activation = self.relu
        elif activation == 'sigmoid':
            self.activation = self.sigmoid
            
        self.params = self.initialize()
        self.cache = {}
        
    def initialize(self):
        np.random.seed(105)
        input_layer = self.sizes[0]
        hidden_layer = self.sizes[1]
        output_layer = self.sizes[2]
        
        params = {
            'W1': np.random.randn(hidden_layer, input_layer) * np.sqrt(2./input_layer),
            'b1': np.zeros((hidden_layer, 1)),
            'W2': np.random.randn(output_layer, hidden_layer) * np.sqrt(2./hidden_layer),
            'b2': np.zeros((output_layer, 1))
        }
        return params
    
    def feed_forward(self, x):
        self.cache['X'] = x
        self.cache['Z1'] = np.dot(self.params['W1'], self.cache['X'].T) + self.params['b1']
        self.cache['A1'] = self.activation(self.cache['Z1'])
        self.cache['Z2'] = np.dot(self.params['W2'], self.cache['A1']) + self.params['b2']
        self.cache['A2'] = self.softmax(self.cache['Z2'])
        return self.cache['A2']
    
    def relu(self, x, derivative=False):
        if derivative:
            return (x > 0).astype(float)
        return np.maximum(0, x)
    
    def sigmoid(self, x, derivative=False):
        if derivative:
            return (np.exp(-x)) / ((np.exp(-x) + 1)**2)
        return 1 / (1 + np.exp(-x))
    
    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
        return exp_z / np.sum(exp_z, axis=0, keepdims=True)
    
    def categorical_crossentropy_loss(self, y, pred):
        m = y.shape[1]
        epsilon = 1e-8  # Add a small epsilon to prevent log(0)
        loss = -(np.mean(y * np.log(pred + epsilon)))
        return loss
    
    def backpropagate(self, y, output):
        m = y.shape[1]
        
        dz2 = output - y
        dW2 = (1./m) * np.dot(dz2, self.cache['A1'].T)
        db2 = (1./m) * np.sum(dz2, axis=1, keepdims=True)
        
        dA1 = np.dot(self.params['W2'].T, dz2)
        dz1 = dA1 * self.activation(self.cache['Z1'], derivative=True)
        dW1 = (1./m) * np.dot(dz1, self.cache['X'])
        db1 = (1./m) * np.sum(dz1, axis=1, keepdims=True) 
        
        self.grads = {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2}
        return self.grads
    
    def optimize(self):
        if self.optimizer == 'sgd':
            for key in self.params:
                self.params[key] -= self.lr * self.grads[key]
                
        elif self.optimizer == 'momentum':
            for key in self.params:
                self.momentum_opt[key] = (self.beta * self.momentum_opt[key] + (1. - self.beta) * self.grads[key])
                self.params[key] -= self.lr * self.momentum_opt[key]
                
    def fit(self, x_train, y_train):
        num_batches = x_train.shape[0] // self.batch_size
        for epoch in range(self.epochs):
            print(f'Epoch {epoch + 1}/{self.epochs}')
            permutation = np.random.permutation(x_train.shape[0])
            x_train_shuffled = x_train[permutation]
            y_train_shuffled = y_train[:, permutation]
            
            for batch in range(num_batches):
                begin = batch * self.batch_size
                end = min(begin + self.batch_size, x_train.shape[0])
                x = x_train_shuffled[begin:end]
                y = y_train_shuffled[:, begin:end]
                y_pred = self.feed_forward(x)
                self.backpropagate(y, y_pred)
                self.optimize()

            y_pred = self.feed_forward(x_train)
            loss = self.categorical_crossentropy_loss(y_train, y_pred)
            self.losses.append(loss)
            print(f'Loss: {loss}')
            print(f'Accuracy: {self.accuracy(y_train, y_pred)}\n')

    def predict(self, x):
        y_pred = self.feed_forward(x)
        return y_pred

    def accuracy(self, y_true, y_pred):
        y_true_labels = np.argmax(y_true, axis=0)
        y_pred_labels = np.argmax(y_pred, axis=0)
        return np.sum(y_true_labels == y_pred_labels) / y_true_labels.shape[0]

# Example usage with the Breast Cancer dataset
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

X, y = datasets.load_iris(return_X_y=True)
encoder = OneHotEncoder(sparse_output=False)
y = encoder.fit_transform(y.reshape(-1, 1))

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=105)
dnn = DeepNeuralNetwork(sizes=[x_train.shape[1], 8, y.shape[1]], optimizer='momentum')
dnn.fit(x_train, y_train.T)
pred = dnn.predict(x_test)

print(f'Test Accuracy: {dnn.accuracy(y_test.T, pred)}')
####### Plotting Training Loss Curve ###################
plt.plot(dnn.losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Curve for Training")
plt.show()
