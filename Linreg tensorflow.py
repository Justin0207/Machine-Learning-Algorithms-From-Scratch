# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 05:51:37 2024

@author: Favour
"""
# import tensorflow as tf
import numpy as np
import tensorflow as tf


from keras.datasets import boston_housing

class SimpleLinearRegression:
    def __init__(self, learning_rate=0.000001, epochs = 1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.w = None
        self.b = None
        
    def fit(self, X, y):
        self.w = tf.Variable(tf.zeros(shape= (X.shape[1])))
        self.b = tf.Variable(0.)
        #self.w.assign([self.var]*X.shape[1])
        
        for i in range(self.epochs):
            print("Epoch: ", i)
            
            self.update(X, y)
        
    def predict(self, x):
        return tf.reduce_sum(self.w * x, 1) + self.b
    
    def mean_squared_error(self, true, predicted):
        return tf.reduce_mean(tf.square(true - predicted))
    
    def update(self, X, y):
        with tf.GradientTape(persistent=True) as g:
            loss = self.mean_squared_error(y, self.predict(X))
            
        print("Loss: ", loss)

        dy_dm = g.gradient(loss, self.w)
        dy_db = g.gradient(loss, self.b)
        
        self.w.assign_sub(self.learning_rate * dy_dm)
        self.b.assign_sub(self.learning_rate * dy_db)
    

         
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()
linear_model = SimpleLinearRegression()
linear_model.fit(x_train, y_train)
pred = linear_model.predict(x_test)
print(pred)
error = linear_model.mean_squared_error(y_test, pred)
print(error)

      