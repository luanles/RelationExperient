# -*- coding: utf-8 -*-

def segment():
    import jieba

    jieba.suggest_freq('沙瑞金', True)
    jieba.suggest_freq('高育良', True)
    jieba.suggest_freq('侯亮平', True)

    with open('E:/in_the_name_of_people.txt', encoding='utf8') as f:
        document = f.read()
        document_cut = jieba.cut(document) # list(document_cut)中会出现'\n'，下一步写回f2时会自动分行，保持原文件格式，很秀~
        result = ' '.join(document_cut)  # 一定要用空格分开，后面的LineSentence才能切成正确的list
        result = result.encode('utf-8')
        with open('E:/in_the_name_of_people_segment.txt', 'wb') as f2:
            f2.write(result)


def generate_word2vec():
    from gensim.models import word2vec
    # 分词后一般会去停用词。但word2vec依赖上下文，而上下文有可能就是停词。因此word2vec可以不去停词。

    # 我们的数据是保存在txt文件中的。每一行对应一个句子（已经分词，以空格隔开），我们可以直接用LineSentence把txt文件转为所需要的格式。
    sentences = word2vec.LineSentence('E:/in_the_name_of_people_segment.txt')
    # sentences = [['first', 'sentence',], ['second', 'sentence'],['haha','sentence']]
    model = word2vec.Word2Vec(sentences, hs=1, min_count=1, window=3, size=100)
    try:
        print(model['你'])
    except KeyError:
        print('该词不在模型中~')

    # 获得词典中的词
    # print(model.wv.vocab.keys())  # model.wv.vocab是个dict(), keys是个[]

    # 保存model(可追加训练)
    model.save('E:/save.txt')  # 打开看不懂...
    # 追加训练
    # model = word2vec.Word2Vec.load(model_path)
    # model.train(more_sentences)

    # 保存model(不能追加训练！！！~)
    model.wv.save_word2vec_format('E:/save_word2vec_format_no_binary.txt', binary=False)  # 形如 '就 -0.13017757 0.022681035 ...'
    # bin模式：model = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=False)
    model.wv.save_word2vec_format('E:/save_word2vec_format_binary.txt', binary=True) # 人看不懂
    # bin模式：model = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=True)



def load_w2v():
    import numpy as np
    '''
    这段代码是处理两个txt
    Word2Vec模型保存的是一份txt，需要用下面的方法！
    add two extra tokens:
        : UNK for unkown tokens
        : BLANK for the max len sentence
    '''
    wordlist = []
    vecs = []

    wordlist.append('BLANK')
    wordlist.extend([word.strip('\n') for word in open('./word_dict.txt')])

    with open('./vecs.txt', encoding='utf8') as f:
        for line in f:
            line = line.strip('\n').split()  # 每一行的0.003307 0.005833 -0.007500 0.002503 ...
            vec = list(map(float, line))  # map把string的0.003307转换成float的0.003307，map返回的是一个迭代器。
            vecs.append(vec)

    vec_dim = len(list(vecs[0]))  # 要先插入已有的vec，才能知道0对应的vec的长度。。（因为要对齐）
    vecs.insert(0, np.zeros(vec_dim))  # 在首行加入全0的vec，对应'BLANK'

    wordlist.append('UNK')   # 给UNK一个随机vec，在尾行加入随机数的vec
    vecs.append(np.random.uniform(low=-1.0, high=1.0, size=vec_dim))

    word2id = {j: i for i, j in enumerate(wordlist)} # id和word构建dict，，直接一个enumerate，就是这么简单粗暴。
    id2word = {i: j for i, j in enumerate(wordlist)}

    return np.array(vecs, dtype=np.float32), word2id, id2word


def load_w2v_from_Word2Vec():
    import numpy as np

    wordlist = []
    vecs = []
    wordlist.append('BLANK')

    with open('E:/save_word2vec_format_no_binary', encoding='utf8') as f:
        flag = 0
        for line in f:
            # 第一行是5095 100，所以忽略
            if flag == 0:
                flag+=1
                continue
            line = line.strip('\n').split()  # 看情况，用啥符号分割，每一行的0.003307 0.005833 -0.007500 0.002503 ...

            wordlist.append(line[0]) # line[0]即 词，后面的都是vec
            vec = list(map(float, line[1:]))  # map把string的0.003307转换成float的0.003307，map返回的是一个迭代器。
            vecs.append(vec)

    vec_dim = len(list(vecs[0]))
    vecs.insert(0, np.zeros(vec_dim))  # 在首行加入全0的vec，对应'BLANK'

    wordlist.append('UNK')  # 在尾行加入随机数的vec,对应'UNK'
    vecs.append(np.random.uniform(low=-1.0, high=1.0, size=vec_dim))

    word2id = {j: i for i, j in enumerate(wordlist)}
    id2word = {i: j for i, j in enumerate(wordlist)}

    return np.array(vecs, dtype=np.float32), word2id, id2word



if __name__ == '__main__':
    # pos1, pos2 = vh()
    # print(pos1.shape)
    vecs, w2i, i2w = load_w2v_from_Word2Vec()
    print(vecs.shape)
    print(w2i)
    print(i2w)