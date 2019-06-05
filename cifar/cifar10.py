#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Lenovo
# @Date:   2019-05-26 16:38:44
# @Last Modified by:   Lenovo
# @Last Modified time: 2019-05-26 18:45:18
import tensorflow as tf
# 函数的输入参数为images,图像的tensor
# 输出个类别的预测标签


def _variable_with_weight_decay(name, shape, stddev, wd):
    dtype = tf.float32
    var = variable_on_cpu(
        name, shape, tf.contrib.layers.xavier_initializer_conv2d(dtype=dtype))
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def inference(images):
    # 建立第一层卷积层
    with tf.variable_scope('conv1') as scope:
        kernel = _variable_with_weight_decay(
            'weight', shape=[5, 5, 3, 64], stddev=5e-2, wd=0.0)
