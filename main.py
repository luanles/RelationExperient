#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Author  : ZHANG Jingzhuo
# @Contact : zhangjingzhuo@pku.edu.cn
# @File    : main.py
# @Time    : 2019-05-14 23:02
# @Desc    : 

# import tensorflow as tf
# from datasets.data1 import Data
# import datasets
# import models
# import time
# import yaml
# import numpy as np
# from sklearn import metrics

# class AttrDict(dict): # 澍哥的这个方法很骚，可以不用config['num_epoch']了，直接config.num_epoch
#     def __init__(self, *args, **kwargs):
#         super(AttrDict, self).__init__(*args, **kwargs)
#         self.__dict__ = self
#
#
# class RelationExtraction():
#
#     def __init__(self):
#         self.read_config()
#         self.sess = tf.Session()
#         # # self.logger = logger
#
#     def read_config(self):
#         config_dict = yaml.load(open('./config.yaml', 'r', encoding='utf-8'))
#         config = AttrDict(config_dict)
#         self.config = config
#
#     def train(self):
#
#         epoch_num = self.config.epoch_num
#
#         # 这里导入Data类！
#         DataModel = getattr(datasets, 'Data')
#         train_data = DataModel(self.config.data_path, data_type='train')
#         # train_data_loader = DataLoader(train_data, opt.batch_size, shuffle=True, num_workers=opt.num_workers,
#         #                                collate_fn=collate_fn)
#         valid_data = DataModel(self.config.data_path, data_type='valid')
#
#         # test_data = DataModel(self.config.data_path, data_type='test')
#         # test_data_loader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers,
#         #                               collate_fn=collate_fn)
#         print('train datas: {}; valid datas: {}'.format(len(train_data), len(valid_data)))
#
#         result = bert_as_feature(querys=, gpu_devices=[0,1,2], data_root='/root/documents/bert_service/chinese_L-12_H-768_A-12/')
#
#         # 这里导入Model类！
#         model = getattr(models, 'Model')(train_data.n_tag, self.config)
#         # models = Model(datas.n_tag, self.config)
#         self.sess.run(model.init)  # 别忘加run init...
#         batch_size = 32
#         for epoch in range(epoch_num):
#             print('>>> new epoch=', epoch)
#             for itera in range(len(train_data)//batch_size):
#
#                 # 这俩东西里面都是32个句子
#                 yield_generator = train_data.generate_batch_data_random(batch_size=32)
#                 x, y = train_data.get_next_batch(yield_generator)
#                 # (32, 261, 768)
#                 text_embed_matrix = np.array([self.bc.encode(line)  for line in x])
#                 tag_matrix = np.array(y)  # (32, 261) 被dict成数字了
#
#                 feed_dict = {model.inputs: text_embed_matrix, model.labels: tag_matrix}
#
#                 self.sess.run(model.train_op, feed_dict=feed_dict)
#
#
#                 print('{} itera={}'.format(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())),itera))
#             loss = self.sess.run(model.loss, feed_dict=feed_dict)
#             transition_params = self.sess.run(model.transition_params, feed_dict=feed_dict)
#             a,p,r,f = self.predict(epoch,itera, len(train_data)//batch_size, model,transition_params, valid_data)
#             print('>>> {} epoch:{} train loss: {} accuracy:{} precision:{} recall:{} F1:{}'.format(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())),
#                                                                                                                    epoch,
#                                                                                                                     loss,
#                                                                                                   round(a,4),round(p,4),round(r,4),round(f,4)))
#             print('>>> finish epoch=', epoch)
#             print('prepare to write result...')
#             self.predict_label(model, transition_params)
#
#
#
# if __name__ == '__main__':
#     relationExtraction = RelationExtraction()
#     relationExtraction.train()


import collections
import codecs
import os
import sys
import time
from sklearn import metrics
import tensorflow as tf
import numpy as np
from models.bert import tokenization
from models.model_bert import *
from datasets.data import Data

'''
This script is based on https://github.com/cjymz886/text_bert_cnn and https://github.com/google-research/bert.
'''


class TextConfig():
    seq_length = 170  # max length of sentence
    num_labels = 2  # number of labels

    num_filters = 128  # number of convolution kernel
    filter_sizes = [2, 3, 4]  # size of convolution kernel
    hidden_dim = 128  # number of fully_connected layer units

    keep_prob = 0.5  # droppout
    lr = 5e-5  # learning rate
    lr_decay = 0.9  # learning rate decay
    clip = 5.0  # gradient clipping threshold

    is_training = True  # is _training
    use_one_hot_embeddings = False  # use_one_hot_embeddings

    num_epochs = 64  # epochs
    batch_size = 32  # batch_size
    print_per_batch = 200  # print result
    require_improvement = 1000  # stop training if no inporement over 1000 global_step

    output_dir = './result'
    data_dir = './corpus/BC5CDR/'  # the path of input_data file
    training_data = data_dir + 'train.txt'
    dev_data = data_dir + 'dev.txt'
    test_data = data_dir + 'test.txt'

    BERT_BASE_DIR = '/root/documents/bert_service/chinese_L-12_H-768_A-12/'
    vocab_file = BERT_BASE_DIR + 'vocab.txt'  # the path of vocab file
    bert_config_file = BERT_BASE_DIR + 'bert_config.json'  # the path of bert_cofig file
    init_checkpoint = BERT_BASE_DIR + 'bert_model.ckpt'  # the path of bert model


def evaluate(sess, dev_data):
    '''Calculate the average loss and accuracy of validation/test data in batch form. '''
    data_len = 0
    total_loss = 0.0
    total_acc = 0.0
    for batch_ids, batch_mask, batch_segment, batch_label in batch_iter(dev_data, config.batch_size):
        batch_len = len(batch_ids)
        data_len += batch_len
        feed_dict = feed_data(batch_ids, batch_mask,
                              batch_segment, batch_label, 1.0)
        loss, acc = sess.run([model.loss, model.acc], feed_dict=feed_dict)
        total_loss += loss * batch_len
        total_acc += acc * batch_len
    return total_loss / data_len, total_acc / data_len


def feed_data(batch_ids, batch_mask, batch_segment, batch_label, keep_prob):
    '''Data for text_model construction. '''
    feed_dict = {
        model.input_ids: np.array(batch_ids),
        model.input_mask: np.array(batch_mask),
        model.segment_ids: np.array(batch_segment),
        model.labels: np.array(batch_label),
        model.keep_prob: keep_prob
    }
    return feed_dict


def train():
    '''Train the text_bert_cnn model. '''
    tensorboard_dir = os.path.join(config.output_dir, "tensorboard/BC5CDR")
    save_dir = os.path.join(config.output_dir, "checkpoints/BC5CDR")
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, 'best_validation')

    start_time = time.time()

    tf.logging.info("*****************Loading training data*****************")
    train_examples = Data().get_train_examples(config.training_data)
    train_data = convert_examples_to_features(
        train_examples, label_list, config.seq_length, tokenizer)

    tf.logging.info("*****************Loading dev data*****************")
    dev_examples = Data().get_dev_examples(config.dev_data)
    dev_data = convert_examples_to_features(
        dev_examples, label_list, config.seq_length, tokenizer)

    tf.logging.info("Time cost: %.3f seconds...\n" %
                    (time.time() - start_time))

    tf.logging.info("Building session and restore bert_model...\n")
    session = tf.Session()
    saver = tf.train.Saver()
    session.run(tf.global_variables_initializer())

    tf.summary.scalar("loss", model.loss)
    tf.summary.scalar("accuracy", model.acc)
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(tensorboard_dir)
    writer.add_graph(session.graph)
    optimistic_restore(session, config.init_checkpoint)

    tf.logging.info('Training and evaluating...\n')
    best_acc = 0
    last_improved = 0  # record global_step at best_val_accuracy
    flag = False

    for epoch in range(config.num_epochs):
        batch_train = batch_iter(train_data, config.batch_size)
        start = time.time()
        tf.logging.info('Epoch:%d' % (epoch + 1))
        # 3_list, 1_label
        for batch_ids, batch_mask, batch_segment, batch_label in batch_train:
            feed_dict = feed_data(batch_ids, batch_mask,
                                  batch_segment, batch_label, config.keep_prob)
            _, global_step, train_summaries, train_loss, train_accuracy = session.run(
                [model.optim, model.global_step, merged_summary, model.loss, model.acc], feed_dict=feed_dict)
            if global_step % config.print_per_batch == 0:
                end = time.time()
                val_loss, val_accuracy = evaluate(session, dev_data)
                merged_acc = (train_accuracy + val_accuracy) / 2
                if merged_acc > best_acc:
                    saver.save(session, save_path)
                    best_acc = merged_acc
                    last_improved = global_step
                    improved_str = '*'
                else:
                    improved_str = ''
                tf.logging.info(
                    "step: {},train loss: {:.3f}, train accuracy: {:.3f}, val loss: {:.3f}, val accuracy: {:.3f},training speed: {:.3f}sec/batch {}".format(
                        global_step, train_loss, train_accuracy, val_loss, val_accuracy,
                        (end - start) / config.print_per_batch, improved_str))
                start = time.time()

            if global_step - last_improved > config.require_improvement:
                tf.logging.info(
                    "No optimization over 1500 steps, stop training")
                flag = True
                break
        if flag:
            break
        config.lr *= config.lr_decay


def test():
    '''testing'''
    save_dir = os.path.join(config.output_dir, "checkpoints/BC5CDR")
    save_path = os.path.join(save_dir, 'best_validation')

    if not os.path.exists(save_dir):
        tf.logging.info("maybe you don't train")
        exit()

    tf.logging.info("*****************Loading testing data*****************")
    test_examples = Data().get_test_examples(config.test_data)
    test_data = convert_examples_to_features(
        test_examples, label_list, config.seq_length, tokenizer)

    input_ids, input_mask, segment_ids = [], [], []

    for features in test_data:
        input_ids.append(features['input_ids'])
        input_mask.append(features['input_mask'])
        segment_ids.append(features['segment_ids'])

    config.is_training = False
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess=session, save_path=save_path)

    tf.logging.info('Testing...')
    test_loss, test_accuracy = evaluate(session, test_data)
    msg = 'Test Loss: {0:>6.2}, Test Acc: {1:>7.2%}'
    tf.logging.info(msg.format(test_loss, test_accuracy))

    batch_size = config.batch_size
    data_len = len(test_data)
    num_batch = int((data_len - 1) / batch_size) + 1
    y_test_cls = [features['label_ids'] for features in test_data]
    y_pred_cls = np.zeros(shape=data_len, dtype=np.int32)

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        feed_dict = {
            model.input_ids: np.array(input_ids[start_id:end_id]),
            model.input_mask: np.array(input_mask[start_id:end_id]),
            model.segment_ids: np.array(segment_ids[start_id:end_id]),
            model.keep_prob: 1.0,
        }
        y_pred_cls[start_id:end_id] = session.run(
            model.y_pred_cls, feed_dict=feed_dict)

    # evaluate
    tf.logging.info("Precision, Recall and F1-Score...")
    tf.logging.info(metrics.classification_report(
        y_test_cls, y_pred_cls, target_names=label_list))

    tf.logging.info("Confusion Matrix...")
    cm = metrics.confusion_matrix(y_test_cls, y_pred_cls)
    tf.logging.info(cm)


if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] not in ['train', 'test']:
        raise ValueError("""usage: python run_BC5CDR.py [train / test]""")

    tf.logging.set_verbosity(tf.logging.INFO)
    config = TextConfig()

    label_list = Data().get_labels()  #

    tokenizer = tokenization.CharTokenizer(
        vocab_file=config.vocab_file)  # 需要更换成中文的Tokenizer，找Git项目

    model = TextCNN(config)
    train()
    # if sys.argv[1] == 'train':
    #     train()
    # elif sys.argv[1] == 'test':
    #     test()
    # else:
    #     exit()
