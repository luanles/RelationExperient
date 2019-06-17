# coding=utf-8

import tensorflow as tf
from models.fork_bert_master import tokenization, modeling
import os
import numpy as np
# fork_bert_master: https://github.com/InsaneLife/bert


# def bad_foo(item, my_list=[]):
# 当你给这些参数赋值为list，dictionary，函数参数的初值只能被计算一次（在函数定义的时间里）
# 非要这么用的话，用None，如下
def bert_as_feature(querys=None, gpu_devices=None,data_root='/root/documents/bert_service/chinese_L-12_H-768_A-12/'):
    if gpu_devices is None:
        gpu_devices = [0]
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(lambda x:str(x), gpu_devices))

    # 配置文件
    bert_config_file = data_root + 'bert_config.json'
    bert_config = modeling.BertConfig.from_json_file(bert_config_file)
    init_checkpoint = data_root + 'bert_model.ckpt'
    bert_vocab_file = data_root + 'vocab.txt'

    # graph
    input_ids = tf.placeholder(tf.int32, shape=[None, None], name='input_ids')
    input_mask = tf.placeholder(tf.int32, shape=[None, None], name='input_masks')
    segment_ids = tf.placeholder(tf.int32, shape=[None, None], name='segment_ids')

    # 初始化BERT
    model = modeling.BertModel(
        config=bert_config,
        is_training=False,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=False)

    # 加载bert模型
    tvars = tf.trainable_variables()
    (assignment, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
    tf.train.init_from_checkpoint(init_checkpoint, assignment)
    # 获取最后一层和倒数第二层。
    encoder_last_layer = model.get_sequence_output()
    # encoder_last2_layer = model.all_encoder_layers[-2]

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        token = tokenization.CharTokenizer(vocab_file=bert_vocab_file)
        querys_last = []
        for query in querys:
            # query = 'Jack,请回答1988, UNwant\u00E9d,running'
            split_tokens = token.tokenize(query)
            word_ids = token.convert_tokens_to_ids(split_tokens)
            word_mask = [1] * len(word_ids)
            word_segment_ids = [0] * len(word_ids)
            fd = {input_ids: [word_ids], input_mask: [word_mask], segment_ids: [word_segment_ids]}
            # last, last2 = sess.run([encoder_last_layer, encoder_last2_layer], feed_dict=fd)
            # print('last shape:{}, last2 shape: {}'.format(last.shape, last2.shape))
            last = sess.run([encoder_last_layer], feed_dict=fd) # 单个query的(1, 26, 768)
            print('last shape:{}'.format(np.array(last).shape))
            querys_last.append(last[0]) # [0]是为了取(26, 768)
            # 如果总共3句，最终querys_last.shape=(3, 26, 768)

    return querys_last