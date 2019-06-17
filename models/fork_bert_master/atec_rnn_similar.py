# coding=utf-8

import tensorflow as tf
import tokenization, modeling, optimization
import os
import collections
from general_utils import Progbar
# import tensorflow.keras.utils.Progbar as Progbar
from atec_rnn_config import Config
import sklearn as sk
from sklearn.metrics import roc_auc_score
from sklearn import metrics
import shutil
import time
import random


class SimilarModel(object):
    def __init__(self, config, train_size):
        self.learning_rate = config.learning_rate
        self.num_labels = config.n_classes  # num of class
        self.logger = config.logger
        self.config = config
        self.num_train_steps = int(train_size / config.train_batch_size * config.nepochs)
        self.num_warmup_steps = int(self.num_train_steps * config.warmup_proportion)

    def initialize_session(self):
        self.logger.info("Initializing tf session")
        global_config = tf.ConfigProto()
        global_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=global_config)
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.tables_initializer())
        self.saver = tf.train.Saver()

    def restore_session(self, dir_model):
        self.logger.info("Reloading the latest trained model...")
        if not os.path.exists(dir_model):
            os.makedirs(dir_model)
        self.saver.restore(self.sess, dir_model)

    def save_session(self):
        if not os.path.exists(self.config.dir_model):
            os.makedirs(self.config.dir_model)
        self.saver.save(self.sess, self.config.dir_model)

    def close_session(self):
        self.sess.close()

    def add_summary(self):
        self.merged = tf.summary.merge_all()
        self.file_writer = tf.summary.FileWriter(self.config.log_dir, self.sess.graph)

    # input_ids, input_mask, segment_ids, labels,
    def add_placeholder(self):
        self.text1_word_ids = tf.placeholder(tf.int32, shape=[None, None], name='text1_word_ids')
        self.text1_char_ids = tf.placeholder(dtype=tf.int32, shape=[None, None, None], name="text1_char_ids")
        self.text1_lengths = tf.placeholder(tf.int32, shape=[None], name='text1_lengths')
        self.text1_word_lengths = tf.placeholder(tf.int32, shape=[None, None], name="text1_word_lengths")
        self.text2_word_ids = tf.placeholder(tf.int32, shape=[None, None], name='text2_word_ids')
        self.text2_char_ids = tf.placeholder(dtype=tf.int32, shape=[None, None, None], name="text2_char_ids")
        self.text2_lengths = tf.placeholder(tf.int32, shape=[None], name='text2_lengths')
        self.text2_word_lengths = tf.placeholder(tf.int32, shape=[None, None], name="text2_word_lengths")
        self.labels = tf.placeholder(tf.int32, shape=[None], name='labels')
        self.dropout = tf.placeholder(dtype=tf.float32, shape=[], name="dropout")

    def add_embedding_layer(self):
        with tf.variable_scope('word_embedding_layer'):
            if self.config.embeddings is None:
                _word_embeddings = tf.get_variable(name='_word_embeddings', dtype=tf.int32,
                                                   shape=[self.config.nwords, self.config.dim_word])
            else:
                _word_embeddings = tf.Variable(self.config.embeddings, dtype=tf.int32,
                                               shape=[self.config.nwords, self.config.dim_word])
            text1_word_embeddings = tf.nn.embedding_lookup(_word_embeddings, self.text1_word_ids)
            text2_word_embeddings = tf.nn.embedding_lookup(_word_embeddings, self.text2_word_ids)

        with tf.variable_scope('char_embedding_layer'):
            if self.config.use_chars:
                _char_embeddings = tf.get_variable(name='_char_embeddings', dtype=tf.int32,
                                                   shape=[self.config.nchars, self.config.dim_char])
                text1_char_embeddings = tf.nn.embedding_lookup(_char_embeddings, self.text1_char_ids,
                                                               name="char_embeddings")
                text2_char_embeddings = tf.nn.embedding_lookup(_char_embeddings, self.text2_char_ids,
                                                               name="char_embeddings")
                # bs, sentence_len, word_len
                s1, s2 = tf.shape(text1_char_embeddings), tf.shape(text2_char_embeddings)
                text1_char_embeddings = tf.reshape(_char_embeddings,
                                                   shape=[s1[0] * s1[1], s1[-2], self.config.dim_char])
                text2_char_embeddings = tf.reshape(_char_embeddings,
                                                   shape=[s2[0] * s2[1], s2[-2], self.config.dim_char])
                text1_word_lengths = tf.reshape(self.text1_word_lengths, shape=[s1[0] * s1[1]])
                text2_word_lengths = tf.reshape(self.text2_word_lengths, shape=[s2[0] * s2[1]])
                # bi Rnn
                cell_fw = tf.nn.rnn_cell.GRUCell(self.config.hidden_size_char, reuse=tf.AUTO_REUSE)
                stacked_gru_fw = tf.nn.rnn_cell.MultiRNNCell([cell_fw], state_is_tuple=True)
                cell_bw = tf.nn.rnn_cell.GRUCell(self.config.hidden_size_char, reuse=tf.AUTO_REUSE)
                stacked_gru_bw = tf.nn.rnn_cell.MultiRNNCell([cell_bw], state_is_tuple=True)
                _, (text1_output_fw, text1_output_bw) = tf.nn.bidirectional_dynamic_rnn(stacked_gru_fw, stacked_gru_bw,
                                                                                        text1_char_embeddings,
                                                                                        sequence_length=text1_word_lengths,
                                                                                        dtype=tf.float32)
                _, (text2_output_fw, text2_output_bw) = tf.nn.bidirectional_dynamic_rnn(stacked_gru_fw, stacked_gru_bw,
                                                                                        text2_char_embeddings,
                                                                                        sequence_length=text2_word_lengths,
                                                                                        dtype=tf.float32)
                text1_output = tf.concat([text1_output_fw, text1_output_bw], axis=-1)
                text1_output = tf.reshape(text1_output, shape=[s1[0], s1[1], 2 * self.config.hidden_size_char])
                text1_word_embeddings = tf.concat([text1_word_embeddings, text1_output], axis=-1)
                self.text1_word_embeddings = tf.nn.dropout(text1_word_embeddings, self.dropout)
                text2_output = tf.concat([text2_output_fw, text2_output_bw], axis=-1)
                text2_output = tf.reshape(text2_output, shape=[s2[0], s2[1], 2 * self.config.hidden_size_char])
                text2_word_embeddings = tf.concat([text2_word_embeddings, text2_output], axis=-1)
                self.text2_word_embeddings = tf.nn.dropout(text2_word_embeddings, self.dropout)

    def add_simlar_layer(self):
        with tf.variable_scope("bi_rnn"):
            # bi Rnn
            cell_fw = tf.nn.rnn_cell.GRUCell(self.config.hidden_size_gru, reuse=tf.AUTO_REUSE)
            stacked_gru_fw = tf.nn.rnn_cell.MultiRNNCell([cell_fw], state_is_tuple=True)
            cell_bw = tf.nn.rnn_cell.GRUCell(self.config.hidden_size_gru, reuse=tf.AUTO_REUSE)
            stacked_gru_bw = tf.nn.rnn_cell.MultiRNNCell([cell_bw], state_is_tuple=True)
            text1_state_output, text1_final_state = tf.nn.bidirectional_dynamic_rnn(stacked_gru_fw,
                                                                                    stacked_gru_bw,
                                                                                    self.text1_word_embeddings,
                                                                                    sequence_length=self.text1_lengths,
                                                                                    dtype=tf.float32)
            self.text1_state_output = tf.concat(text1_state_output, axis=-1)
            self.text1_final_state = tf.concat(text1_final_state, axis=-1)
            text2_state_output, text2_final_state = tf.nn.bidirectional_dynamic_rnn(stacked_gru_fw, stacked_gru_bw,
                                                                                    self.text2_word_embeddings,
                                                                                    sequence_length=self.text2_lengths,
                                                                                    dtype=tf.float32)
            self.text2_state_output = tf.concat(text2_state_output, axis=-1)
            self.text2_final_state = tf.concat(text2_final_state, axis=-1)

            # todo: add attention layer

        with tf.variable_scope("cosine_similar"):
            # Cosine similarity
            # text1_norm = sqrt(sum(each x^2))
            text1_norm = tf.sqrt(tf.reduce_sum(tf.square(self.text1_final_state), 1, True))
            text2_norm = tf.sqrt(tf.reduce_sum(tf.square(self.text2_final_state), 1, True))

            prod = tf.reduce_sum(tf.multiply(self.text1_final_state, self.text2_final_state), 1, True)
            norm_prod = tf.multiply(text1_norm, text2_norm)

            # cos_sim_raw = query * doc / (||query|| * ||doc||), [bs]
            cos_sim_raw = tf.truediv(prod, norm_prod)
            # gamma = 20
            self.cos_sim = cos_sim_raw
        with tf.variable_scope("manhattan_distance"):
            self.diff = tf.reduce_sum(tf.abs(tf.subtract(self.text1_final_state, self.text1_final_state)), axis=1)  #
            self.similarity = tf.exp(-1.0 * self.diff)
        # MSE
        with tf.variable_scope("loss"):
            diff = tf.subtract(self.similarity, self.labels - 1.0) / 4.0  # 32
            self.loss = tf.square(diff)  # (batch_size,)
            self.cost = tf.reduce_mean(self.loss)  # (1,)