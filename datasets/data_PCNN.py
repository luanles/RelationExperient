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
    def __init__(self, train=True):
        # if train:
        #     path = os.path.join(root_path, 'train/')
        #     print('loading train data')
        # else:
        #     path = os.path.join(root_path, 'test/')
        #     print('loading test data')

        e2id = np.load('../source/e2id.npy').item()  # 这里一定要加item()！
        rel2id = np.load('../source/rel2id.npy').item()

        self.bags_feature = []
        self.bags_label = []
        with codecs.open('../source/preprocess.txt', 'r', 'utf8') as reader:
            while 1:
                line = reader.readline().strip()
                if not line:
                    break
                section_list = line.split('\t')
                e1_id = int(e2id[section_list[0]])
                e2_id = int(e2id[section_list[1]])
                label_id = int(rel2id[section_list[2]])
                self.bags_label.append(label_id)

                num = int(section_list[-1])
                this_bag_feature = [e1_id, e2_id]
                # TODO 加入e1,e2的描述信息，到包feature中

                this_bag_sentence = []
                for i in range(num):
                    sent = reader.readline().strip()  # sent是真实的每个bag中的每个句子
                    this_bag_sentence.append(sent)
                this_bag_feature.append(this_bag_sentence)
                self.bags_feature.append(this_bag_feature)

                # self.bags_feature =
                # [[3, 6, ['考来烯胺、矿物油、新霉素、硫糖铝能干扰本品中维生素A的吸收。']],
                #  [8, 1, ['抗酸药（如氢氧化铝）可影响本品中维生素A的吸收，故不应同服。']],
                #  [9, 1, ['链霉素可提高血浆维生素A的浓度。']],
                #  [0, 5, ['阿司匹林不应与含有大量镁、钙的药物合用。以免引起高镁、高钙血症。']],
                #  [5, 7, ['抗酸药（如氢氧化铝）可影响本品中维生素A的吸收，故不应同服。']],
                #  [2, 1, ['大量维生素A与抗凝药（如香豆素）同服，可导致凝血酶原降低。', '与香豆素同用，可增加维生素A的吸收，增加其肝内贮存量，加速利用和降低毒性，但大量香豆素可消耗维生素A在体内的贮存。']],
                #  [4, 7, ['高锰酸钾如与其他药物（青霉素）同时使用可能会发生药物相互作用，详情请咨询医师或药师。']]]

                # self.bags_label =
                # [3, 1, 1, 0, 3, 2, 4]

                # self.labels = np.load('../source/e2id.npy')
                # self.x = np.load('../source/rel2id.npy')
                # # 好关键哦！！把两个融合了。。
                # self.x = zip(self.x, self.labels)

        print('data finish !')

    def __getitem__(self, idx):
        assert idx < len(self.bags_feature)
        return self.bags_feature[idx]

    def __len__(self):
        return len(self.bags_label)

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

    def get_labels(self):
        """See base class."""
        return [0, 1, 2, 3, 4]

    """read file"""

    def _read_file(self, input_file):
        with codecs.open(input_file, "r", encoding='utf-8') as f:
            lines = []
            for line in f.readlines():
                try:
                    line = line[:-1].split('\t')
                    lines.append(line)
                except:
                    pass
            np.random.shuffle(lines)
            return lines

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if len(line) != 4:
                print(line)
            guid = "%s-%s-%s" % (set_type,
                                 tokenization.convert_to_unicode(line[0]), str(i))
            target = tokenization.convert_to_unicode(line[1])
            text = tokenization.convert_to_unicode(line[2])
            label = tokenization.convert_to_unicode(line[3])
            examples.append(
                InputExample(guid=guid, text_a=target, text_b=text, label=label))
        return examples


class Data_Preprocess:
    def __init__(self):
        print('preprocess start....')
        e_set = set()
        df = pd.DataFrame(columns=['entity1', 'entity2', 'relation', 'text'])
        with codecs.open('../source/train.txt', 'r', 'utf8') as reader:
            for line in reader:
                section_list = (line.strip()).split(' ')
                assert len(section_list) == 4, '这行' + line.strip() + '源数据不是4个元素啊，请检查一下'
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
        rel2id = {'NA': 0, '禁忌合用': 1, '谨慎合用': 2, '不推荐合用': 3, '关注': 4}
        # print(rel2id)

        # 判别e1,e2相同时，rel是否相同，不同返回所有rel值，要提醒洗数据
        df_groupby_e1_e2 = df.groupby(by=['entity1', 'entity2'], as_index=False).apply(lambda x: set(x['relation']))
        for i in range(len(df_groupby_e1_e2)):
            if len(df_groupby_e1_e2.iloc[i]) > 1:
                raise KeyError('第{}行的两个实体{}对应的关系出现重复：{}，请及时到源文件中修改！'.format(i, df_groupby_e1_e2.index[i],
                                                                            df_groupby_e1_e2.iloc[i]))
        # print(df_groupby_e1_e2)

        # 正式进行groupby,把相同e1,e2的句子合并到一个包里
        df_groupby_e1_e2_rel = df.groupby(by=['entity1', 'entity2', 'relation']).apply(lambda x: list(x['text']))
        print(df_groupby_e1_e2_rel)

        with codecs.open('../source/preprocess.txt', 'w', 'utf8') as f:  # 'w'：只写（如果文件不存在，则自动创建文件）
            for i in range(len(df_groupby_e1_e2_rel)):
                e1, e2, rel = df_groupby_e1_e2_rel.index[i]
                text_list = df_groupby_e1_e2_rel.iloc[i]
                f.write(e1 + '\t' + e2 + '\t' + rel + '\t' + str(len(text_list)) + '\n')
                for line in text_list:
                    f.write(line + '\n')

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
    # d = Data_Preprocess()
    d = Data(train=True)
    print('bags_feature: ', d.bags_feature)
    print('bags_label: ', d.bags_label)
    print('e2id: ', np.load('../source/e2id.npy').item())
    print('rel2id: ', np.load('../source/rel2id.npy').item())
