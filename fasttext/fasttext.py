#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Lenovo
# @Date:   2019-06-03 19:49:45
# @Last Modified by:   Lenovo
# @Last Modified time: 2019-06-03 21:25:34
import tensorflow as tf


class FastTextConfig(object):

    """fasttext 参数配置"""
    embedding_dim = 200
    seq_length = 600
    num_classes = 2
    vocab_size = 5000
    learning_rate = 1e-3
    learning_decay_rate = 0.1
    learning_decay_step = 100
    epochs = 20
    dropout_keep_prob = 0.5


class fasttext(object):
    # fasttext模型

    def __init__(self, config):
        self.config = config
        self.input_x = tf.placeholder(
            tf.int32, [None, self.config.seq_length], name='input_x')
        self.input_y = tf.placeholder(
            tf.float32, [None, self.config.num_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float, name='keep_prob')

    def fast(self):
        with tf.name_scope("embedding"):
            self.embedding = tf.get_variable(
                "embedding", [self.vocab_size, self.embedding_dim])
            embedding_inputs = tf.nn.embedding_lookup(
                self.embedding, self.input_x)

        with tf.name_scope("dropout"):
            dropout_output = tf.nn.dropout(
                embedding_inputs, self.dropout_keep_prob)
        # 对词向量进行平均
        with tf.name_scope("average"):
            mean_sentence = tf.reduce_mean(dropout_output, axis=1)

        # 输出层
        with tf.name_scope("score"):
            self.logits = tf.layers.dense(
                mean_sentence, self.num_classes, name="dense_layer")
            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)  # 预测类别
        # 损失函数
        self.loss = tf.nn.softmax_cross_entropy_with_logits(
            logits=self.logits, labels=self.input_y)
        # 优化函数
        self.global_step = tf.train.get_or_create_global_step()
        learning_rate = tf.train.exponential_decay(
            self.config.learning_rate, self.global_step,
            self.config.learning_decay_step, self.config.learning_decay_rate, staircase=True)
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
        with tf.name_scope("accuracy"):
            # 准确率
            correct_pred = tf.equal(
                tf.argmax(self.input_y, 1), self.y_pred_cls)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
