# Copyright (c) 2016-2017, Deogtae Kim & DTWARE Inc. All rights reserved.
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = ","
# del os.environ["CUDA_VISIBLE_DEVICES"]

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

tf.reset_default_graph()
tf.set_random_seed(107)

## 데이터 수집

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

## 데이터 시각화, 전처리

print(type(mnist.train))
print(dir(mnist))
print(type(mnist.train.images), type(mnist.train.labels))
print(mnist.train.images.shape)
print(mnist.train.images[0].shape)
print(mnist.train.images[0])
print(mnist.train.labels.shape)
print(mnist.train.labels[0].shape)
print(mnist.train.labels[0])
print(mnist.train.num_examples, mnist.train.images.shape, mnist.train.labels.shape)
print(mnist.validation.num_examples, mnist.validation.images.shape, mnist.validation.labels.shape)
print(mnist.test.num_examples, mnist.test.images.shape, mnist.test.labels.shape)

fig, axes = plt.subplots(5, 5)
for i in range(5):
    for j in range(5):
        axes[i,j].axis('off')
        axes[i,j].imshow(mnist.train.images[i*5+j].reshape(28,28))
        axes[i,j].set_title("%d" % np.argmax(mnist.train.labels[i*5+j]))

## 예측 모델 정의: 소프트맥스 회귀 모델

X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
# 각 데이터에 대한 각 분류별 점수
score = tf.matmul(X, W) + b
# 각 데이터에 대한 각 분류별 확률
pred = tf.nn.softmax(score)

## 손실 함수, 정확도, 최적화 함수 정의

cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(pred), reduction_indices=[1]))
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cost)

## 훈련

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
import time
start = time.time()
for epoch in range(50):
    avg_cost = 0
    batch_count = int(mnist.train.num_examples / 100)
    for _ in range(batch_count):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        c, _  = sess.run([cost, train_step], feed_dict={X: batch_xs, Y: batch_ys})
        avg_cost += c / batch_count
    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost), 
          ', validation accuacy = ', '{:.9f}'.format(sess.run(accuracy, feed_dict={X: mnist.validation.images, Y: mnist.validation.labels})))
    
## 모델 평가

print('accuacy = ', '{:.9f}'.format(sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels})))
print("훈련 시간:", time.time() - start)  
sess.close()

## 재훈련: 최적화 알고리듬 변경 (Adam)

train_step = tf.train.AdamOptimizer(0.001).minimize(cost)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
import time
start = time.time()
for epoch in range(50):
    avg_cost = 0
    batch_count = int(mnist.train.num_examples / 100)
    for _ in range(batch_count):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        c, _  = sess.run([cost, train_step], feed_dict={X: batch_xs, Y: batch_ys})
        avg_cost += c / batch_count
    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost), 
          ', validation accuacy = ', '{:.9f}'.format(sess.run(accuracy, feed_dict={X: mnist.validation.images, Y: mnist.validation.labels})))
    
print('accuacy = ', '{:.9f}'.format(sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels})))
print("훈련 시간:", time.time() - start)  

## 모델 평가

print(sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels}))
sess.close()
