# Copyright (c) 2016-2017, Deogtae Kim & DTWARE Inc. All rights reserved.
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = ","
# del os.environ["CUDA_VISIBLE_DEVICES"]

import tensorflow as tf

tf.reset_default_graph()
tf.set_random_seed(107)

## 데이터 수집

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

## 하이퍼 매개변수 설정

learning_rate = 0.001
training_epochs = 15
batch_size = 100

## 예측 모델 정의: 피드 포워드 네트워크 (Feed Forward Network)

X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])

W1 = tf.get_variable("W1", shape=[784, 512],
                     initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([512]))
L1 = tf.matmul(X, W1) + b1

W2 = tf.get_variable("W2", shape=[512, 512],
                     initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([512]))
L2 = tf.matmul(L1, W2) + b2

W3 = tf.get_variable("W3", shape=[512, 512],
                     initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([512]))
L3 = tf.matmul(L2, W3) + b3

W4 = tf.get_variable("W4", shape=[512, 512],
                     initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([512]))
L4 = tf.matmul(L3, W4) + b4

W5 = tf.get_variable("W5", shape=[512, 10],
                     initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random_normal([10]))
# 각 데이터에 대한 각 분류별 점수
score = tf.matmul(L4, W5) + b5
# 각 데이터에 대한 각 분류별 확률
pred = tf.nn.softmax(score)

## 손실 함수, 최적화 함수 정의

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=score))
#cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(pred), reduction_indices=[1])) # 발산하기 쉬움 (nan)
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cost)
#train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

## 훈련

sess = tf.Session()
sess.run(tf.global_variables_initializer())
import time
start = time.time()
for epoch in range(training_epochs):
    avg_cost = 0
    batch_count = int(mnist.train.num_examples / batch_size)
    for _ in range(batch_count):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        c, _  = sess.run([cost, train_step], feed_dict={X: batch_xs, Y: batch_ys})
        avg_cost += c / batch_count
    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost),
          ', accuacy = ', '{:.9f}'.format(sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels})))
print("훈련 시간:", time.time() - start)  

## 모델 평가

print('Accuracy:', sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels}))
sess.close()
