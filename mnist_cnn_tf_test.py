# Copyright (c) 2016-2017, Deogtae Kim & DTWARE Inc. All rights reserved.
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = ","
# del os.environ["CUDA_VISIBLE_DEVICES"]

import tensorflow as tf
import numpy as np

tf.reset_default_graph()
np.random.seed(20171201)
tf.set_random_seed(20171201)

## 데이터 수집

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

## 예측 모델 정의: 합성곱 신경망 (CNN)

# 가중치(weight) 초기화 루틴
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

# 합성곱 계층 및 풀링 계층 생성 루틴
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

# 입력 계층
X = tf.placeholder(tf.float32, [None, 784])
X_image = tf.reshape(X, [-1,28,28,1])
Y = tf.placeholder(tf.float32, [None, 10])

# 합성곱 계층, 풀링 계층 생성
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

h_conv1 = tf.nn.relu(conv2d(X_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# 2번째 합성곱 계층, 풀링 계층 생성
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# 완전 연결 계층 (Fully Connected Layer) 생성
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 출력 계층 생성
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

# 각 데이터에 대한 각 분류별 점수
score = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
# 각 데이터에 대한 각 분류별 확률
pred = tf.nn.softmax(score)

## 손실 함수, 최적화 함수 정의

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=score))
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cost)

## 훈련

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
import time
start = time.time()
for epoch in range(15):
    avg_cost = 0
    batch_size = 100
    batch_count = int(mnist.train.num_examples / batch_size)
    for _ in range(batch_count):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        c, _  = sess.run([cost, train_step], feed_dict={X: batch_xs, Y: batch_ys, keep_prob: 0.5})
        avg_cost += c / batch_count
    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost),
          ', accuacy = ', '{:.9f}'.format(sess.run(accuracy, feed_dict={
            X: mnist.test.images, Y: mnist.test.labels, keep_prob: 1.0})))
print("훈련 시간:", time.time() - start)  


## 모델 평가

print('Accuracy:', sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels, keep_prob: 1.0}))
sess.close()
