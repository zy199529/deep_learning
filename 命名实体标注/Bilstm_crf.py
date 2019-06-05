#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Lenovo
# @Date:   2019-06-04 20:32:11
# @Last Modified by:   Lenovo
# @Last Modified time: 2019-06-04 22:22:24
import tensorflow as tf


class BiLSTM_CRF_Config(object):
    batch_size = 64
    epoch_num = 10
    hidden_dim = 128
    embeddings = 200
    update_embedding =
    CRF =
    dropout_keep_prob = 0.5
    lr = 1e-3
    optimizer =
    clip_grad =
    tag2label =
    vocab = 5000
    suffle =
    model_path =
    summary_path =
    result path =


class BiLSTM_CRF(object):

    def __init__(self, config):
        self.config = config
        self.word_ids = tf.placeholder(
            tf.int32, shape=[None, None], name="word_ids")
        self.labels = tf.placeholder(
            tf.int32, shape=[None, None], name="labels")
        self.sequence_lengths = tf.placeholder(
            tf.int32, shape=[None], name="sequence_lengths")
        self.dropout+pl = tf.placeholder(dtype=tf.float32, shape=[], name="dropout")
        self.lr_pl = tf.placeholder(dtype=tf.float32, shape=[], name="lr")
        self.bilstm_crf()

    def bilstm_crf():
        with tf.name_scope("words"):
            _word_embedding = tf.Variable(
                self.config.embeddings, dtype=tf.float32, trainable=self.config.update_embedding, name="_word_embedding")
            word_embendding = tf.nn.embedding_lookup(
                params=_word_embedding, ids=self.word_ids, name="word_embedding")

        with tf.name_scope("dropout"):
            self.word_embendding = tf.nn.dropout(
                word_embendding, self.dropout_pl)

        with tf.name_scope("bi_lstm"):
            cell_fw = tf.nn.rnn_cell.LSTMCell(self.config.hidden_dim)
            cell_bw = tf.nn.rnn_cell.LSTMCell(self.config.hidden_dim)
            (output_fw_seq, output_bw_seq), _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw, cell_bw=cell_bw, inputs=self.word_embendding, sequence_lengths=self.config.sequence_lengths, dtype=tf.float32)
            output = tf.concat([output_fw_seq, output_bw_seq], axis=1)
            output = tf.nn.dropout(output, self.dropout_pl)
