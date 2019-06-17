#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Author  : ZHANG Jingzhuo
# @Contact : zhangjingzhuo@pku.edu.cn
# @File    : bert.py
# @Time    : 2019-06-06 21:32
# @Desc    : 

import sys
import codecs
import numpy as np
import tensorflow as tf
from models.fork_bert_master.bert_as_feature import bert_as_feature
# fork_bert_master: https://github.com/InsaneLife/bert


class BertModel:


    def __init__(self):
        print('This demo demonstrates how to load the pre-trained model and extract the sentence embedding with' \
              ' pooling.')

    def bert_model(self):
        # ['考来烯胺、矿物油、新霉素、硫糖铝能干扰本品中维生素A的吸收。', '糖铝能干扰本品']
        # 3
        #TODO 传入的是bag的那种形式，一个包里的[]句子，然后进行bert得出向量，然后做tf的Attention，softmax分类，得到最终一个loss
        # self.inputs = tf.placeholder(tf.float32, shape=[None, 261, 768])  # 261个字，输入m=261个cell里
        # self.labels = tf.placeholder(tf.int32, shape=[None, 261])  # 是 int32 ！！
        # 再用self把需要暴露的变量给self一下

        self.inputs = tf.placeholder(tf.float32, shape=[None, None])  # 261个字，输入m=261个cell里
        self.labels = tf.placeholder(tf.int32, shape=[None])  # 是 int32 ！！






