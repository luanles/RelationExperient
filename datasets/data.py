#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Author  : ZHANG Jingzhuo
# @Contact : zhangjingzhuo@pku.edu.cn
# @File    : data.py
# @Time    : 2019-06-07 23:31
# @Desc    : 

import codecs, os
import numpy as np
import pandas as pd
from models.bert import tokenization


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence.
          text_b: string. The untokenized text of the second sequence.
          label: string. The label of the example. This should be specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class Data(object):
    """Processor for the BC5CDR corpus."""

    def get_train_examples(self, training_data):
        return self._create_examples(
            self._read_file(training_data), "train")

    def get_dev_examples(self, dev_data):
        return self._create_examples(
            self._read_file(dev_data), "dev")

    def get_test_examples(self, test_data):
        return self._create_examples(
            self._read_file(test_data), "test")

    # TODO 如果要改标签种类，这里要记得变更！！！
    def get_labels(self):
        """See base class."""
        return ['0', '1', '2', '3', '4']

    def _read_file(self, input_file):
        rel2id = np.load('../source/rel2id.npy').item()  # 这里一定要加item()！

        bags = []
        with codecs.open('../source/preprocess.txt', 'r', 'utf8') as reader:
            while 1:
                line = reader.readline().strip()
                if not line:
                    break
                section_list = line.split('\t')

                label_id = rel2id[section_list[2]]

                # TODO 加入e1,e2的描述信息，到包feature中

                one_bag_sentence = []
                num = int(section_list[-1])
                for i in range(num):
                    sent = reader.readline().strip()  # sent是真实的每个bag中的每个句子
                    one_bag_sentence.append(sent)
                one_bag = [label_id, one_bag_sentence]
                bags.append(one_bag)
                np.random.shuffle(bags)

        # bags= [[3, ['新霉素、硫糖铝能干扰维生素A的吸收。', '与香豆素同用，可增加维生素A的吸收，增加其肝内贮存量]],...]
        return bags

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):

            guid = "%s-%s" % (set_type, str(i))
            texts = []
            for text in line[1]:  # line[1] is a list !
                texts.append(tokenization.convert_to_unicode(text))
            label = tokenization.convert_to_unicode(line[0])  # label位于第一个位置
            examples.append(
                InputExample(guid=guid, text_a=texts, label=label))  # text_a is a list !
        return examples

class Data_Preprocess:
    def __init__(self):
        print('preprocess start....')
        e_set = set()
        df = pd.DataFrame(columns=['entity1','entity2','relation','text'])
        with codecs.open('../source/train.txt', 'r', 'utf8') as reader:
            for line in reader:
                section_list = (line.strip()).split(' ')
                assert len(section_list) == 4, '这行'+line.strip()+'源数据不是4个元素啊，请检查一下'
                # 如果condition为false，那么raise一个AssertionError出来。

                e_set.add(section_list[0])
                e_set.add(section_list[1])
                new_line = pd.DataFrame({'entity1': section_list[0],
                                    'entity2': section_list[1],
                                    'relation': section_list[2],
                                    'text': section_list[3]}, index=[0])

                # "-------在原数据框df1最后一行新增一行，用append方法------------"
                df = df.append(new_line, ignore_index=True)  # ignore_index=True,表示不按原来的索引，从0开始自动递增

        e2id = dict()
        for index, e in enumerate(e_set):
            e2id[e] = index
        # print(e2id)
        rel2id = {'NA': '0', '禁忌合用': '1', '谨慎合用': '2', '不推荐合用': '3', '关注': '4'}
        # print(rel2id)

        # 判别e1,e2相同时，rel是否相同，不同返回所有rel值，要提醒洗数据
        df_groupby_e1_e2 = df.groupby(by=['entity1', 'entity2'],as_index=False).apply(lambda x: set(x['relation']))
        for i in range(len(df_groupby_e1_e2)):
            if len(df_groupby_e1_e2.iloc[i]) > 1:
                raise KeyError('第{}行的两个实体{}对应的关系出现重复：{}，请及时到源文件中修改！'.format(i, df_groupby_e1_e2.index[i],
                                                                  df_groupby_e1_e2.iloc[i]))
        # print(df_groupby_e1_e2)

        # 正式进行groupby,把相同e1,e2的句子合并到一个包里
        df_groupby_e1_e2_rel = df.groupby(by=['entity1','entity2','relation']).apply(lambda x:list(x['text']))
        print(df_groupby_e1_e2_rel)

        with codecs.open('../source/preprocess.txt', 'w', 'utf8') as f: # 'w'：只写（如果文件不存在，则自动创建文件）
            for i in range(len(df_groupby_e1_e2_rel)):
                e1, e2, rel = df_groupby_e1_e2_rel.index[i]
                text_list = df_groupby_e1_e2_rel.iloc[i]
                f.write(e1+'\t'+e2+'\t'+rel+'\t'+str(len(text_list))+'\n')
                for line in text_list:
                    f.write(line+'\n')

        # 构成一个 e1,e2,rel的 'text \n text \n text' 的dict()，并计数有几句
        # 构建e2id, rel2id + entity描述信息（说明书里找） || 构建e2id, rel2id, vec, vocab
        # bert: 写入重构文件，形式e1, e2, rel, num 换行 num行句子 || pcnn: e1_id, e2_id, rel_id, num 换行 num行vocab_id句子
        # 读文件，看内存，可分批读，可一起读到内存中
        # bert编码每一条句子，然后做句子级别Attention，得到bag_vec
        #

        np.save('../source/e2id.npy', e2id)
        np.save('../source/rel2id.npy', rel2id)

        print('preprocess finish!')



if __name__ == '__main__':
    # Data_Preprocess()
    # d = Data()
    # print('bags_feature: ',d.bags)
    # print('bags_label: ',d.bags_label)
    # print('e2id: ',np.load('../source/e2id.npy').item())
    # print('rel2id: ', np.load('../source/rel2id.npy').item())
    print(Data().get_train_examples(training_data='1'))
