#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Lenovo
# @Date:   2019-05-05 15:47:17
# @Last Modified by:   Lenovo
# @Last Modified time: 2019-05-05 16:25:54
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# 创建x,x是一个占位符（placeholder），代表待识别的图片
x = tf.placeholder(tf.float32, [None, 784])
# W是Softmax模型的参数，将一个784维的输入转换为一个10维的输出
# 在Tensorflow中，变量的参数用tf.Variable表示
W = tf.Variable(tf.zeros([784, 10]))
# b是偏置项
b = tf.Variable(tf.zeros([10]))
# y表示模型的输出
y = tf.nn.softmax(tf.matmul(x, W)+b)
# y_表示实际图像标签，占位符表示
y_ = tf.placeholder(tf.float32, [None, 10])
# 构造交叉熵
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y)))
# 梯度下降法，优化损失 学习率0.01
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
# 创建一个session。只有在session中才能运行优化步骤train_step
sess = tf.InteractiveSession()
# 运行之前必须要初始化所有变量，分配内存
tf.global_variables_initializer().run()
for _ in range(1000):
    # 取出mnist.train中100个训练数据
    # batch_xs代表图像数据（100，784），batch_ys是（100，10)的实际标签
    batch_xs, batch_ys = mnist.train.next_batch(100)
    # 传入会话中
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
"""
每次不使用全部训练数据，而是每次提取100个数据进行训练，共1000次
"""
# 正确的预测结果
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
# 计算预测准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(
    sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
