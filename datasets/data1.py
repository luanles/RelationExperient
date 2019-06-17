#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Author  : ZHANG Jingzhuo
# @Contact : zhangjingzhuo@pku.edu.cn
# @File    : data1.py
# @Time    : 2019-05-14 23:50
# @Desc    : 
from bert_serving.client import BertClient
import numpy as np
import re
import random
import os

# 先去打开bert_service.server！
# bert-serving-start -model_dir ~/project/pycharm_project/chinese_L-12_H-768_A-12/ -num_worker=4

bc = BertClient()

class Data:
    '''
    return 只发生在向外界传递数据时，比如给main 1batch的数据时，其余的文件读取、预处理、全放在类里面，用self传递~
    总，后面要加迭代器batch输出数据

    x = pad_sentence (32019, 149, 200)

    y = pad_label (32019, 149)
    [array([3, 30, 13, 30, 3, 34, 41, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=object),
    array([30, 13, 3, 41, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=object),

    '''

    def __init__(self, config, root_path ,data_type = 'train'):
        self.config = config
        self.data_type = data_type
        if data_type == 'train':
            path = os.path.join(root_path, 'train_pos.txt')
            print('loading train datas...')

        elif data_type == 'valid':
            path = os.path.join(root_path, 'val_pos.txt')
            print('loading valid datas...')
        else:
            path = os.path.join(root_path, 'test2_seg.txt')
            print('loading test datas...')

        self.labels = np.load(path + 'labels.npy')
        self.x = np.load(path + 'bags_feature.npy')
        # 好关键哦！！把两个融合了。。
        self.x = zip(self.x, self.labels)

        print('loading finish')


    def __getitem__(self, idx):
        assert idx < len(self.data5)
        return self.data5[idx]

    def __len__(self):
        return len(self.data5)


    def batch_genernator(self, batch_size):  # 定义batch数据生成器

        start = 0
        while True:

            if start + batch_size > len(pad_label_data4):
                data_batch_x = np.hstack([lispad_data4[start:], pad_data4[:(start + batch_size) % len(pad_data4)]])
                data_batch_y = np.hstack(
                    [pad_label_data4[start:], pad_label_data4[:(start + batch_size) % len(pad_label_data4)]])

            else:
                data_batch_x = pad_data4[start: start + batch_size]
                data_batch_y = pad_label_data4[start: start + batch_size]

            start = (start + batch_size) % len(pad_label_data4)
            data_batch_bert_x = []
            for line in data_batch_x:
                data_batch_bert_x.append(bc.encode(list(line)))

            yield data_batch_bert_x, data_batch_y



class Data_Load:

    def __init__(self):

        print('loading start....')
        self.w2v, self.word2id, self.id2word = self.load_w2v()
        self.p1_2v, self.p2_2v = self.load_p2v()

        np.save(os.path.join(self.root_path, 'w2v.npy'), self.w2v)
        np.save(os.path.join(self.root_path, 'p1_2v.npy'), self.p1_2v)
        np.save(os.path.join(self.root_path, 'p2_2v.npy'), self.p2_2v)




    def preprocess(self):

        if self.data_type == 'train' or self.data_type == 'valid':

            np.save(os.path.join(self.root_path, 'w2v.npy'), self.w2v)
            data1 = []
            with open(self.path, encoding='utf8') as f:
                for line in f:



    all_weight_distribution = []

    for vec in vectors:
        one_vec_dot_result = np.dot(vec, vectors.transpose())
        #     print(one_vec_dot_result.shape)
        one_vec_dot_result -= np.max(one_vec_dot_result)
        exp_one_vec_result = np.exp(one_vec_dot_result)
        exp_sum_result = np.sum(exp_one_vec_result)  # 所有e的和
        one_itme_weight_distribution = exp_one_vec_result / exp_sum_result  # 得到一条句子的注意力概率分布
        all_weight_distribution.append(one_itme_weight_distribution)
    # all_weight_distribution.shape (7, 7) 第1行是第1句对应的七个注意力weight

    all_attention_matrix = []
    for one_itme_weight_distribution in all_weight_distribution:
        c = []
        for (w, line) in zip(one_itme_weight_distribution, vectors):
            c.append(line * w)
        #         print(np.array(c).shape)
        whole = np.sum(c, axis=0)
        all_attention_matrix.append(whole)
        # print(np.array(all_attention_matrix).shape)  # (7,768) 得到自注意力机制后的新向量组
        # 自注意力机制，下面要加入普通注意力机制。

    relation_vectors = bc.encode(['鱼肝油乳 ||| 维生素E'])  # (1, 768) 两个实体，输出一个关系向量，用来做关系分类！

    score_list = np.dot(all_attention_matrix, np.array(relation_vectors).transpose())
    # print(score_list.shape) # (7,1)
    score_list -= np.max(score_list)
    exp_list = np.exp(score_list)
    exp_sum = np.sum(exp_list)
    sentence_atten = exp_list / exp_sum
    # print(np.array(sentence_atten)) # (7,1)

    sentence_atten = sentence_atten.reshape([7])  # (7,)

    bag_vec = []
    for w, line in zip(sentence_atten, all_attention_matrix):
        bag_vec.append(w * line)
    bag_vec = np.sum(bag_vec, axis=0)
    bag_vec = bag_vec.reshape([1, -1])  # (1, 768) 该7个句子经过处理后的一条包向量

    label_vec = [[0, 1, 0, 0, 0]]

    return bag_vec, label_vec



if __name__ == "__main__":
    data = Data_Load('./dataset/FilterNYT/')