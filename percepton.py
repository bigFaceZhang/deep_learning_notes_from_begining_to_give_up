#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/7/24 0024 20:50
# @Author  : LiuXf
# @File    : percepton.py
# @Software: PyCharm Community Edition

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division

import numpy as np
import tensorflow as tf

# datasets
from sklearn.datasets import load_breast_cancer
from sklearn.cross_validation import train_test_split

# ##### classes
class percepton_model_np():
    def sign(self, x):
        return np.where(x>=0, 1, -1)
    def fit(self, x, y, iter_num=10000, batch_size=1, learning_rate=0.01, stop_value=0.85):
        graph_0 = tf.Graph()
        
        self.train_sample_num = x.shape[0]
        self.variable_num = x.shape[1]# independent variables' num
        # w: the layers' weight value
        self.w = np.random.uniform(-0.5, 1, self.variable_num)
        # b: the layers' bias value
        self.bias = np.random.uniform(-0.5, 1, 1)
        
        for i in xrange(iter_num):
            alpha = np.dot(x, self.w) + self.bias
            y_predict = self.sign(alpha)
            error = y_predict * y
            self.accurate = np.equal(y_predict,y).mean() # np.equal(self.predict(x),y).mean()
            if self.accurate < stop_value:
                error_index = error == -1
                batch_index = np.random.choice(np.arange(self.train_sample_num)[error_index],batch_size)
                self.w -= (x[batch_index].transpose() * y_predict[batch_index]).mean(axis=1) * learning_rate
                self.bias -= y_predict[batch_index].mean() * learning_rate
            else:
                break
            if i % (iter_num // 10) ==1:
                print("current train dataset accuracy : %f"%np.equal(self.predict(x),y).mean())
    def predict(self, x):
        alpha = np.dot(x, self.w) + self.bias
        y_predict = self.sign(alpha)
        return y_predict
        
class percepton_model_tf():
    def fit(self, x_train, y_train, iter_num=5000, batch_size=1, learning_rate=0.01):
        self.graph = tf.Graph()
        train_sample_num = x_train.shape[0]
        self.variable_num = x_train.shape[1]
        with self.graph.as_default():
            x = tf.placeholder(tf.float32, [None, self.variable_num])
            y = tf.placeholder(tf.float32, [None, 1])
            w = tf.Variable(tf.random_normal([self.variable_num,1], dtype=tf.float32, name='weight'))
            bias = tf.Variable(tf.random_normal([1], dtype=tf.float32, name='bias'))
            y_predict = tf.matmul(x,w) + bias # tf.sign(tf.matmul(x,w) + bias)
            
            correct_prediction = tf.equal(tf.sign(y_predict), y)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            
            loss = -tf.where(tf.less_equal(tf.multiply(y_predict, y),0), 
                tf.multiply(y_predict, y), 
                tf.constant(np.repeat(0, batch_size).reshape([batch_size,1]), 
                dtype=tf.float32))
            train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
            # train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
        with tf.Session(graph = self.graph) as sess:
            sess.run(tf.global_variables_initializer())
            feed_dict={x:x_train, y:y_train.reshape([train_sample_num,1])}
            for i in xrange(iter_num):
                batch_index = np.random.choice(np.arange(train_sample_num), batch_size)
                step_dict={x:x_train[batch_index], y:y_train.reshape([train_sample_num,1])[batch_index]}
                sess.run(train_step, feed_dict=step_dict)
                if i % (iter_num // 10) ==0:
                    print("current train dataset accuracy : %f"%sess.run(accuracy, feed_dict=feed_dict))
            print("final train dataset accuracy : %f"%sess.run(accuracy, feed_dict=feed_dict))
            self.w, self.bias = sess.run([w, bias])
    def predict(self, x_):
        return np.sign(np.dot(x_,self.w) + self.bias).reshape(x_.shape[0])
    def test(self, x_, y_):
        print("test dataset accuracy : %f"%np.equal(self.predict(x_),y_).mean())
# ##### data load and process
breast_cancer = load_breast_cancer()
x_data = breast_cancer.data
y_data_0 = breast_cancer.target
y_data_1 = np.where(y_data_0 == 0, -1, y_data_0)

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data_1, test_size=0.3)

# ##### model
model = percepton_model_np()
model.fit(x_train, y_train, iter_num=10000, batch_size=100, learning_rate=0.01, stop_value=0.92)
test_accurate = np.equal(model.predict(x_test),y_test).mean()
print(model.accurate, test_accurate)

# current train dataset accuracy : 0.381910
# current train dataset accuracy : 0.522613
# current train dataset accuracy : 0.776382
# 0.922110552764 0.93567251462

model = percepton_model_tf()
model.fit(x_train, y_train, iter_num=10000, batch_size=100, learning_rate=0.01)
model.test(x_test, y_test)
# current train dataset accuracy : 0.623116
# current train dataset accuracy : 0.801508
# current train dataset accuracy : 0.896985
# current train dataset accuracy : 0.964824
# current train dataset accuracy : 0.957286
# current train dataset accuracy : 0.924623
# current train dataset accuracy : 0.969849
# current train dataset accuracy : 0.969849
# current train dataset accuracy : 0.972362
# current train dataset accuracy : 0.939699
# final train dataset accuracy : 0.952261
# test dataset accuracy : 0.947368
