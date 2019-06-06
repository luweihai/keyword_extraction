# coding=utf-8
# 采用Word2Vec词聚类方法抽取关键词——根据候选关键词的词向量进行聚类分析

import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')  # 忽略警告
import codecs
import numpy as np
import pandas as pd
import jieba
import jieba.posseg
import gensim

# 返回特征词向量
def getWordVecs(wordList, model):
    word_name = []
    word_vectors = []
    for word in wordList:
        word = word.replace('\n', '')
        try:
            if word in model:  # 模型中存在该词的向量表示
                word_name.append(word.encode('utf8'))
                word_vectors.append(model[word])
        except KeyError:
            print("KeyError")
            continue
    words = pd.DataFrame(word_name, columns=['words'])
    vectors = pd.DataFrame(np.array(word_vectors, dtype='float'))
    return pd.concat([words, vectors], axis= 1)

# 数据预处理操作：分词，去停用词，词性筛选
def preprocessing(text, stopkey):
    word_list = []
    pos = ['n', 'nz', 'v', 'vd', 'vn', 'l', 'a', 'd'] # 定义选取的词性
    seg = jieba.posseg.cut(text)  # 分词
    for i in seg :
        if i.word not in stopkey and i.flag in pos: # 去停用词 + 词性筛选
            word_list.append(i.word)
    return word_list

# 根据数据获取候选关键词的词向量
def getAllWordsVecs(data, stopkey, model):
    idList, titleList, abstractList = data['id'], data['title'], data['abstract']
    for index in range(len(idList)):
        id = idList[index]
        title = titleList[index]
        abstract = abstractList[index]
        title_words = preprocessing(title, stopkey)  # 处理标题
        abstract_words = preprocessing(abstract, stopkey) # 处理摘要

        words = np.append(title_words, abstract_words) # 拼接标题和摘要的词向量
        words = list(set(words)) # 数组元素去重
        word_vectors = getWordVecs(words, model)   # 获取候选关键词的词向量表示

        df_word_vectores = pd.DataFrame(word_vectors)
        df_word_vectores.to_csv('result/vectors/wordvecs_' + str(id), index= False)
        print("document ", id, " done")

def main():
    dataFile = "data/sample_data.xlsx"
    data = pd.read_excel(dataFile)
    stopkey = [word.strip() for word in codecs.open("data/stopWord.txt", "rb").readlines()]  # 停用词
    # 词向量模型
    inp = 'wiki.zh.text.vector' # 训练好的 wiki 中文语料词向量
    model = gensim.models.KeyedVectors.load_word2vec_format(inp, binary=False)
    getAllWordsVecs(data, stopkey, model)

if __name__ == '__main__':
    main()









