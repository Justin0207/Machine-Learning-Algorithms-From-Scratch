# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 17:00:33 2024

@author: Favour
"""

import numpy as np
import tensorflow as tf
from keras.datasets import boston_housing
import matplotlib.pyplot as plt

###### Do not use the same learning rate for the different optimizers

losses = []
class TensorLinReg:
    def __init__(self, lr = 0.001, epochs = 10000, momentum = 0.99):
        self.w = None
        self.b = None
        self.lr = lr
        self.epochs = epochs
        self.rho = momentum
        
    def fit(self, X, y, optimizer = 'SGD'):
        n_samples, n_features = X.shape
        self.w = tf.Variable(tf.zeros(shape= n_features))
        self.b = tf.Variable(0.)
        
        if optimizer.lower() == 'nesterov':
            vx = 0.
            vy = 0.
            for i in range(self.epochs):
                print('Epoch {}'.format(i))
                dw, db = self.compute_gradient(X, y)
                old_vx = vx
                old_vy = vy
                vx = self.rho * vx - self.lr * dw
                vy = self.rho * vy - self.lr * db
                self.w.assign_add(-self.rho * old_vx + (1 + self.rho) * vx)
                self.b.assign_add(-self.rho * old_vy + (1 + self.rho) * vy)
                
        elif optimizer.lower() == 'momentum':
            vx = 0.
            vy = 0.
            for i in range(self.epochs):
                print('Epoch {}'.format(i))
                dw, db = self.compute_gradient(X, y)
                vx = self.rho * vx + dw
                vy = self.rho * vy + db
                self.w.assign_sub(self.lr * vx)
                self.b.assign_sub(self.lr * vy)
                
        elif optimizer.lower() == 'adagrad':
            grad_w_squared = 0
            grad_b_squared = 0
            for i in range(self.epochs):
                print('Epoch {}'.format(i))
                dw, db = self.compute_gradient(X, y)
                grad_w_squared += dw * dw 
                grad_b_squared += db * db
                self.w.assign_sub(self.lr * dw/(np.sqrt(grad_w_squared) + 1e-7))
                self.b.assign_sub(self.lr * db/(np.sqrt(grad_b_squared) + 1e-7))
                
        elif optimizer.lower() == 'rmsprop':
            grad_w_squared = 0
            grad_b_squared = 0
            decay_rate = 0.1
            for i in range(self.epochs):
                print('Epoch {}'.format(i))
                dw, db = self.compute_gradient(X, y)
                grad_w_squared = decay_rate * grad_w_squared + (1- decay_rate) * dw * dw 
                grad_b_squared = decay_rate * grad_b_squared + (1- decay_rate) * db * db
                self.w.assign_sub(self.lr * dw/(np.sqrt(grad_w_squared) + 1e-7))
                self.b.assign_sub(self.lr * db/(np.sqrt(grad_b_squared) + 1e-7))
                
        elif optimizer.lower() == 'adam':
            first_moment = 0
            second_moment = 0
            beta1 = 0.9
            beta2 = 0.9
            for i in range(self.epochs):
                print('Epoch {}'.format(i))
                dw, db = self.compute_gradient(X, y)
                first_moment = beta1 * first_moment + (1- beta1) * dw 
                second_moment = beta2 * second_moment + (1- beta2) * dw * dw 
                self.w.assign_sub(self.lr * first_moment/(np.sqrt(second_moment) + 1e-7))
                self.b.assign_sub(self.lr * db)
        elif optimizer.lower() == 'sgd':
           for i in range(self.epochs):
               print('Epoch {}'.format(i))
               dw, db = self.compute_gradient(X, y)
               self.w.assign_sub(self.lr * dw)
               self.b.assign_sub(self.lr * db) 
        else:
            print('Invalid optimizer!!!, choose one of [adam, adagrad, sgd, nesterov, momentum, rmsprop]')
        
        
    def mse(self, true, pred):
        return tf.reduce_mean(tf.square(true - pred))
    
    def compute_gradient(self, X, y):
        with tf.GradientTape(persistent = True) as tape:
            loss = self.mse(y, self.predict(X))
        print('Loss :', loss)
        losses.append(loss)
        dl_dw = tape.gradient(loss, self.w)
        dl_db = tape.gradient(loss, self.b)
        return dl_dw, dl_db
            
    def predict(self, X):
        return tf.reduce_sum(self.w * X, 1) + self.b
    
    
    
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()
linear_model = TensorLinReg()
linear_model.fit(x_train, y_train, optimizer = 'adam')
pred = linear_model.predict(x_test)
print(pred)
error = linear_model.mse(y_test, pred)
print(error)
plt.plot(range(10000), losses)
plt.ylim([min(losses), max(losses)])
plt.show()
