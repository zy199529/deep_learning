#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Lenovo
# @Date:   2019-05-25 11:18:55
# @Last Modified by:   zy19950209
# @Last Modified time: 2019-05-25 13:08:22
import tensorflow as tf
with tf.Session() as sess:
    filename = [
        './mnist_data/A.jpg', './mnist_data/B.jpg', './mnist_data/C.jpg']
    filename_queue = tf.train.string_input_producer(
        filename, shuffle=False, num_epochs=5)
    reader = tf.WholeFileReader()  # 读取所有数据
    key, value = reader.read(filename)
    tf.local_variables_initializer().run()
    threads = tf.train.start_queue_runners(sess=sess)
    i = 0
    while True:
        i = i+1
        image_data = sess.run(value)
        with open('read/test_%d.jpg' % i, 'wb') as f:
            f.write(image_data)
