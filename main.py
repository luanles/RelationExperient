#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Author  : ZHANG Jingzhuo
# @Contact : zhangjingzhuo@pku.edu.cn
# @File    : main.py
# @Time    : 2019-05-14 23:02
# @Desc    : 

import tensorflow as tf
from datasets.data1 import Data
import datasets
import models
import time
import yaml
import numpy as np
from sklearn import metrics

class AttrDict(dict): # 澍哥的这个方法很骚，可以不用config['num_epoch']了，直接config.num_epoch
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class RelationExtraction():

    def __init__(self):
        self.read_config()
        self.sess = tf.Session()
        # # self.logger = logger

    def read_config(self):
        config_dict = yaml.load(open('./config.yaml', 'r', encoding='utf-8'))
        config = AttrDict(config_dict)
        self.config = config

    def train(self):

        epoch_num = self.config.epoch_num

        # 这里导入Data类！
        DataModel = getattr(datasets, 'Data')
        train_data = DataModel(self.config.data_path, data_type='train')
        # train_data_loader = DataLoader(train_data, opt.batch_size, shuffle=True, num_workers=opt.num_workers,
        #                                collate_fn=collate_fn)
        valid_data = DataModel(self.config.data_path, data_type='valid')

        # test_data = DataModel(self.config.data_path, data_type='test')
        # test_data_loader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers,
        #                               collate_fn=collate_fn)
        print('train datas: {}; valid datas: {}'.format(len(train_data), len(valid_data)))

        result = bert_as_feature(querys=, gpu_devices=[0,1,2], data_root='/root/documents/bert_service/chinese_L-12_H-768_A-12/')

        # 这里导入Model类！
        model = getattr(models, 'Model')(train_data.n_tag, self.config)
        # models = Model(datas.n_tag, self.config)
        self.sess.run(model.init)  # 别忘加run init...
        batch_size = 32
        for epoch in range(epoch_num):
            print('>>> new epoch=', epoch)
            for itera in range(len(train_data)//batch_size):

                # 这俩东西里面都是32个句子
                yield_generator = train_data.generate_batch_data_random(batch_size=32)
                x, y = train_data.get_next_batch(yield_generator)
                # (32, 261, 768)
                text_embed_matrix = np.array([self.bc.encode(line)  for line in x])
                tag_matrix = np.array(y)  # (32, 261) 被dict成数字了

                feed_dict = {model.inputs: text_embed_matrix, model.labels: tag_matrix}

                self.sess.run(model.train_op, feed_dict=feed_dict)


                print('{} itera={}'.format(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())),itera))
            loss = self.sess.run(model.loss, feed_dict=feed_dict)
            transition_params = self.sess.run(model.transition_params, feed_dict=feed_dict)
            a,p,r,f = self.predict(epoch,itera, len(train_data)//batch_size, model,transition_params, valid_data)
            print('>>> {} epoch:{} train loss: {} accuracy:{} precision:{} recall:{} F1:{}'.format(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())),
                                                                                                                   epoch,
                                                                                                                    loss,
                                                                                                  round(a,4),round(p,4),round(r,4),round(f,4)))
            print('>>> finish epoch=', epoch)
            print('prepare to write result...')
            self.predict_label(model, transition_params)



if __name__ == '__main__':
    relationExtraction = RelationExtraction()
    relationExtraction.train()