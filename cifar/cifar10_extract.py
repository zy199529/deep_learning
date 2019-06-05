#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: zy19950209
# @Date:   2019-05-25 13:30:55
# @Last Modified by:   Lenovo
# @Last Modified time: 2019-05-26 18:45:19
import os
import scipy
import tensorflow as tf
IMAGE_SIZE = 32
NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000


def inputs_origin(data_dir):  # 读取cifar的图片，将其转换为输入数据，全部传入队列
    filenames = [os.path.join(data_dir, 'data_batch_%d' % i)
                 for i in range(1, 6)]
    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file:'+f)
    filename_queue = tf.train.string_input_producer(filenames)
    # 转换为图像
    read_input = read_cifar10(filename_queue)
    # 将图片转换为实数
    reshaped_image = tf.cast(read_input.uint8image, tf.float32)
    return reshaped_image


def read_cifar10(filename_queue):
    class CIFAR10Record(object):
        pass
    result = CIFAR10Record()
    label_bytes = 1
    result.height = 32
    result.width = 32
    result.depth = 3
    image_bytes = result.height*result.width*result.depth
    record_bytes = label_bytes+image_bytes
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    result.key, value = reader.read(filename_queue)
    # 将原来编码为字符串类型的变量重新变回来
    record_bytes = tf.decode_raw(value, tf.uint8)
    # 读取标签
    result.label = tf.cast(
        tf.strided_slice(record_bytes, [0], [label_bytes]), tf.int32)
    # 将图片格式转换为颜色，长，宽
    depth_major = tf.reshape(tf.strided_slice(record_bytes, [label_bytes], [
                             label_bytes+image_bytes]), [result.depth, result.height, result.width])
    # 颜色，长，宽变为，长，宽，颜色
    result.uint8image = tf.transpose(depth_major, [1, 2, 0])
    return result
# 数据增强方法


def _generate_image_and_label_batch(image, label, min_queue_examples, batch_size, shuffle):
        # 线程数
    num_preprocess_threads = 8
    if shuffle:
        images, label_batch = tf.train.shuffle_batch(
            [image,
                label], batch_size=batch_size, num_threads=num_preprocess_threads,
            capacity=min_queue_examples+3*batch_size, min_after_dequeue=min_queue_examples)
    else:
        images, label_batch = tf.train.batch(
            [image,
                label], batch_size=batch_size, num_threads=num_preprocess_threads,
            capacity=min_queue_examples+3*batch_size)
    tf.summary.image('images', images)
    return images, tf.reshape(label_batch, [batch_size])


def distorted_inputs(data_dir, batch_size):
    filenames = [os.path.join(data_dir, 'data_batch_%d' % i)
                 for i in range(1, 6)]
    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file:'+f)
    # 建立队列
    filename_queue = tf.train.string_input_producer(filenames)
    # 开启队列
    read_input = read_cifar10(filename_queue)
    reshaped_image = tf.cast(read_input.uint8image, tf.float32)
    height = IMAGE_SIZE
    weight = IMAGE_SIZE
    # 随机裁剪图片
    distorted_image = tf.random_crop(reshaped_image, [height, weight, 3])
    # 随机旋转图片
    distorted_image = tf.random_flip_left_right(distorted_image)
    # 亮度变换
    distorted_image = tf.image.random_brightness(distorted_image, max_delta=63)
    # 对比度变换
    distorted_image = tf.image.random_contrast(
        distorted_image, lower=0.2, upper=1.8)
    # 标准化
    float_image = tf.image.per_image_standardization(distorted_image)
    float_image.set_shape([height, weight, 3])
    read_input.label.set_shape([1])
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(
        NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN*min_fraction_of_examples_in_queue)
    print('Filling queue with %d CIFAR images before starting to train. '
          'This will take a few minutes.' % min_queue_examples)
    return _generate_image_and_label_batch(float_image, read_input.label, min_queue_examples, batch_size, shuffle=True)
    
if __name__ == '__main__':
    with tf.Session() as sess:
        reshaped_image = inputs_origin('./cifar-10-batches-py')
        # 启动填充队列
        threads = tf.train.start_queue_runners(sess=sess)
        # 定义全局变量
        sess.run(tf.global_variables_initializer())
        # 定义路径
        if not os.path.exists('./raw/'):
            os.makedirs('./raw/')
        # 保存图片
        for i in range(30):
            image_array = sess.run(reshaped_image)
            scipy.misc.toimage(image_array).save('./raw/%d.jpg' % i)
