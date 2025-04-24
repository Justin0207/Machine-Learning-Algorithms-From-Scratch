# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 05:08:35 2024

@author: Favour
"""

import numpy as np
import tensorflow as tf



class TensorLogReg:
    def __init__(self, epochs= 1000, lr= 0.0001, optimizer = 'SGD'):
        self.w = None
        self.b = None
        self.lr = lr
        self.epochs = epochs
        
    def fit(self, X, y, optimizer = 'SGD'):
        n_samples, n_features = X.shape
        self.w = tf.Variable(tf.zeros(shape= n_features))
        self.b = tf.Variable(0.)
        
        if optimizer.lower() == 'nesterov':
            vx = 0.
            vy = 0.
            for i in range(self.epochs):
                print('Epoch {}'.format(i))
                acc = accuracy(y, self.predict(X))
                print('Accuracy: ', acc)
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
                acc = accuracy(y, self.predict(X))
                print('Accuracy: ', acc)
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
                acc = accuracy(y, self.predict(X))
                print('Accuracy: ', acc)
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
                acc = accuracy(y, self.predict(X))
                print('Accuracy: ', acc)
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
                acc = accuracy(y, self.predict(X))
                print('Accuracy: ', acc)
                dw, db = self.compute_gradient(X, y)
                first_moment = beta1 * first_moment + (1- beta1) * dw 
                second_moment = beta2 * second_moment + (1- beta2) * dw * dw 
                self.w.assign_sub(self.lr * first_moment/(np.sqrt(second_moment) + 1e-7))
                self.b.assign_sub(self.lr * db)
        elif optimizer.lower() == 'sgd':
           for i in range(self.epochs):
               print('Epoch {}'.format(i))
               acc = accuracy(y, self.predict(X))
               print('Accuracy: ', acc)
               dw, db = self.compute_gradient(X, y)
               self.w.assign_sub(self.lr * dw)
               self.b.assign_sub(self.lr * db) 
        else:
            print('Invalid optimizer!!!, choose one of [adam, adagrad, sgd, nesterov, momentum, rmsprop]')
    def inference(self, X):
        logits = tf.reduce_sum(self.w * X, 1) + self.b
        return tf.math.sigmoid(logits)
        
        
    def predict_class(self, probability, thresh=0.5):
        return tf.cast(probability > 0.5, tf.float32)
        
    def cross_entropy(self, y, y_pred):
        loss = tf.reduce_mean(y*tf.math.log((y_pred)) + (1-y) * tf.math.log(1-(y_pred)))
        return -loss
    def compute_gradient(self, X,y):
        with tf.GradientTape(persistent=True) as tape:
            loss = self.cross_entropy(y, self.inference(X))
        print('Loss: {}\n'.format(loss))
        dl_dw = tape.gradient(loss, self.w)
        dl_db = tape.gradient(loss, self.b)
        return dl_dw, dl_db
    
    def predict(self, X):
        pred = self.inference(X)
        return self.predict_class(pred)
def accuracy(y_test, y_pred):
    check_equal = tf.cast(y_pred == y_test,tf.float32)
    acc_val = tf.reduce_mean(check_equal)
    return acc_val

    
from sklearn import datasets
from sklearn.model_selection import train_test_split
X, y = datasets.load_breast_cancer(return_X_y = True)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 105)
lr = TensorLogReg()
print(lr.w)
print(lr.b)
lr.fit(x_train, y_train, optimizer = 'adam')
pred = lr.predict(x_test)
print(pred)
print(accuracy(y_test, pred))
        