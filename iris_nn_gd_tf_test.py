# Copyright (c) 2016-2017, Deogtae Kim & DTWARE Inc. All rights reserved.
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = ","
# del os.environ["CUDA_VISIBLE_DEVICES"]

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import model_selection
from sklearn import metrics


tf.reset_default_graph()
tf.set_random_seed(107)

## 데이터 수집

dataset = datasets.load_iris()

## 데이터를 훈련 데이터와 테스트 데이터로 분할

X_data = dataset.data
y_data = dataset.target
from sklearn.preprocessing import label_binarize
y_data = np.asarray(label_binarize(y_data, [0, 1, 2]), dtype=np.float64)
X_train, X_test, y_train, y_test = model_selection.train_test_split(X_data, y_data, test_size = .3, random_state=101, stratify=y_data)
X_train.shape, X_test.shape, y_train.shape, y_test.shape
print(np.unique(y_train, return_counts=True))

## 예측 모델 정의: 소프트맥스 회귀 모델

X = tf.placeholder(tf.float32, [None, 4])
Y = tf.placeholder(tf.float32, [None, 3])
W1 = tf.Variable(tf.random_uniform([4,100], -1.0, 1.0))
b1 = tf.Variable(tf.zeros([100]))
h = tf.nn.relu(tf.matmul(X, W1) + b1)

W2 = tf.Variable(tf.random_uniform([100,3], -1.0, 1.0))
b2 = tf.Variable(tf.zeros([3]))
# 각 데이터에 대한 각 분류별 점수
score = tf.matmul(h, W2) + b2

pred = tf.nn.softmax(score)

## 손실 함수, 정확도, 최적화 함수 정의

cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(pred), reduction_indices=[1]))
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

## 훈련

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
import time
start = time.time()
for epoch in range(1000):
    c, _  = sess.run([cost, train_step], feed_dict={X: X_train, Y: y_train})
    if (epoch+1) % 100 == 0:
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(c), 
              ', accuacy = ', '{:.9f}'.format(sess.run(accuracy, feed_dict={X: X_test, Y: y_test})))
print("훈련 시간:", time.time() - start)  

## 모델 평가
pred2 = sess.run(pred, feed_dict={X: X_train, Y: y_train})
print(pred2[0:4,])
print("혼돈 행렬:", metrics.confusion_matrix(np.argmax(y_train, axis=1), np.argmax(pred2, axis=1)), sep="\n")
sess.close()
