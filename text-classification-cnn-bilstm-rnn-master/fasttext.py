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
    num_epochs = 20
    dropout_keep_prob = 0.5
    batch_size = 64  # 每批训练大小
    print_per_batch = 100  # 每多少轮输出一次结果
    save_per_batch = 10  # 每多少轮存入tensorboard
    train_filename = './data/cnews/train.txt'  # 训练数据train data
    test_filename = './data/cnews/test.txt'  # 测试数据test data
    val_filename = './data/cnews/test.txt'  # 验证集validation data
    vocab_filename = './data/cnews/vector_word.txt'  # vocabulary词汇集6000
    vector_word_filename = './data/cnews/vector_word.txt'  # vector_word trained by word2vec 经过word2vec训练后的词向量
    vector_word_npz = './data/cnews/vector_word.npz'  # save vector_word to numpy file



class Fasttext(object):
    # fasttext模型

    def __init__(self, config):
        self.config = config
        self.input_x = tf.placeholder(
            tf.int32, [None, self.config.seq_length], name='input_x')
        self.input_y = tf.placeholder(
            tf.float32, [None, self.config.num_classes], name='input_y')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropoutKeepProb")
        self.fast()
    def fast(self):
        with tf.name_scope("embedding"):
            self.embedding = tf.get_variable(
                "embedding", [self.config.vocab_size, self.config.embedding_dim])
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
                mean_sentence, self.config.num_classes, name="dense_layer")
            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)  # 预测类别
        # 损失函数
        with tf.name_scope("optimize"):
            # 损失函数，交叉熵
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy)
            # 优化器
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

        with tf.name_scope("accuracy"):
            # 准确率
            correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
