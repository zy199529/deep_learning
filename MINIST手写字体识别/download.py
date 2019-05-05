#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Lenovo
# @Date:   2019-05-05 14:06:33
# @Last Modified by:   Lenovo
# @Last Modified time: 2019-05-05 15:36:37
# coding:utf-8
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import scipy.misc
import os
# 引入模块
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# 把原始图片存在'MNIST_data/raw/'
save_dir = 'MNIST_data/raw/'
if os.path.exists(save_dir) is False:
    os.makedirs(save_dir)
# 查看数据集大小
# (55000, 784)
# (55000, 10)
# (5000, 784)
# (5000, 10)
# (10000, 784)
# (10000, 10)
print(mnist.train.images.shape)
print(mnist.train.labels.shape)
print(mnist.validation.images.shape)
print(mnist.validation.labels.shape)
print(mnist.test.images.shape)
print(mnist.test.labels.shape)

# 打印出第0张图片向量
print(mnist.train.images[0, :])
# 保存前20张图片
for i in range(20):
    # mnist.train.images[i, :] 表示第i张图片
    image_array = mnist.train.images[i, :]
    # 将784个向量还原为28*28维的图像
    image_array = image_array.reshape(28, 28)
    # 保存
    filename = save_dir+'mnist_train_%d.jpg' % i
    # 将image_array保存为图片
    scipy.misc.toimage(image_array, cmin=0.0, cmax=1.0).save(filename)
    one_hot_label = mnist.train.labels[i, :]
    # 通过np.argmax,直接得到获得原始的label
    label = np.argmax(one_hot_label)
    print("mnist_train_%d.jpg label:%d" % (i, label))
