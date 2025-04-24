# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 14:27:23 2024

@author: Favour
"""

import numpy as np
import matplotlib.pyplot as plt
class DeepNeuralNetwork:
    def __init__(self, sizes, activation='relu', optimizer='momentum', epochs=10000, batch_size=32, lr=0.0001, beta=0.9):
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
                'b2': np.zeros((sizes[2], 1)),
                'W3': np.zeros((sizes[3], sizes[2])),
                'b3': np.zeros((sizes[3], 1))
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
        hidden_layer1 = self.sizes[1]
        hidden_layer2 = self.sizes[2]
        output_layer = self.sizes[3]
        
        params = {
            'W1': np.random.randn(hidden_layer1, input_layer) * np.sqrt(2./input_layer),
            'b1': np.zeros((hidden_layer1, 1)),
            'W2': np.random.randn(hidden_layer2, hidden_layer1) * np.sqrt(2./hidden_layer1),
            'b2': np.zeros((hidden_layer2, 1)),
            'W3': np.random.randn(output_layer, hidden_layer2) * np.sqrt(2./hidden_layer2),
            'b3': np.zeros((output_layer, 1))
        }
        return params
    
    def feed_forward(self, x):
        self.cache['X'] = x
        self.cache['Z1'] = np.dot(self.params['W1'], self.cache['X'].T) + self.params['b1']
        self.cache['A1'] = self.activation(self.cache['Z1'])
        self.cache['Z2'] = np.dot(self.params['W2'], self.cache['A1']) + self.params['b2']
        self.cache['A2'] = self.activation(self.cache['Z2'])
        self.cache['Z3'] = np.dot(self.params['W3'], self.cache['A2']) + self.params['b3']
        self.cache['A3'] = self.sigmoid(self.cache['Z3'])
        return self.cache['A3']
    
    def relu(self, x, derivative=False):
        if derivative:
            x = np.where(x < 0, 0, x)
            x = np.where(x >= 0, 1, x)
            return x
        return np.maximum(0, x)
    
    def sigmoid(self, x, derivative=False):
        if derivative:
            return (np.exp(-x)) / ((np.exp(-x) + 1)**2)
        return 1 / (1 + np.exp(-x))
    
    def binary_crossentropy_loss(self, y, pred):
        m = y.shape[0]
        epsilon = 1e-8  # Add a small epsilon to prevent log(0)
        loss = -(1/m) * np.sum(y*np.log(pred + epsilon) + (1-y)*np.log(1-pred + epsilon))
        return loss
    
    def backpropagate(self, y, output):
        current_batch_size = y.shape[0]
        
        dz3 = output - y
        dW3 = (1./current_batch_size) * np.dot(dz3, self.cache['A2'].T)
        db3 = (1./current_batch_size) * np.sum(dz3, axis=1, keepdims=True)
        
        dA2 = np.dot(self.params['W3'].T, dz3)
        dz2 = dA2 * self.activation(self.cache['Z2'], derivative=True)
        dW2 = (1./current_batch_size) * np.dot(dz2, self.cache['A1'].T)
        db2 = (1./current_batch_size) * np.sum(dz2, axis=1, keepdims=True) 
        
        dA1 = np.dot(self.params['W2'].T, dz2)
        dz1 = dA1 * self.activation(self.cache['Z1'], derivative=True)
        dW1 = (1./current_batch_size) * np.dot(dz1, self.cache['X'])
        db1 = (1./current_batch_size) * np.sum(dz1, axis=1, keepdims=True)         
        
        self.grads = {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2, 'W3': dW3, 'b3': db3}
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
            print('Epoch', epoch + 1)
            permutation = np.random.permutation(x_train.shape[0])
            x_train_shuffled = x_train[permutation]
            y_train_shuffled = y_train[permutation]
            
            for batch in range(num_batches):
                begin = batch * self.batch_size
                end = min(begin + self.batch_size, x_train.shape[0])
                x = x_train_shuffled[begin:end]
                y = y_train_shuffled[begin:end]
                y_pred = self.feed_forward(x)
                self.backpropagate(y, y_pred)
                self.optimize()
            y_pred = self.feed_forward(x_train)
            loss = self.binary_crossentropy_loss(y_train, y_pred)
            print('Loss', loss)
            self.losses.append(loss)
            print('Accuracy {}\n'.format(accuracy(y_train, (y_pred > 0.5).astype(int))))
    def predict(self, x):
        return np.where(self.feed_forward(x) > 0.5, 1, 0)

def accuracy(y_test, y_pred):
    acc = np.sum(y_pred == y_test)/len(y_test)
    return acc
from sklearn import datasets
from sklearn.model_selection import train_test_split
X, y = datasets.load_breast_cancer(return_X_y = True)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 105)
lr = DeepNeuralNetwork(sizes = [x_train.shape[1], 8, 32, 1])
lr.fit(x_train, y_train)
pred = lr.predict(x_test)
print(pred)
print(accuracy(y_test, pred))
plt.plot(lr.losses)
plt.xlabel("Iteration")
plt.ylabel("Log Loss")
plt.title("Loss Curve for Training")
plt.show()