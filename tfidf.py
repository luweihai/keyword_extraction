# coding=utf-8

import codecs
import pandas as pd
import numpy as np
import jieba.posseg
import jieba.analyse
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

"""
    TF-IDF权重：
        1、CountVectorizer 构建词频矩阵
        2、TfidfTransformer 构建 tfidf 权值计算
        3、文本的关键词
        4、对应的 tfidf 矩阵
"""

# 数据预处理操作：分词，去停用词，词性筛选
def preprocessing(text, stopkey):
    word_list = []
    pos = ['n', 'nz', 'v', 'vd', 'vn', 'l', 'a', 'd'] # 定义选取的词性
    seg = jieba.posseg.cut(text)  # 分词
    for i in seg :
        if i.word not in stopkey and i.flag in pos: # 去停用词 + 词性筛选
            word_list.append(i.word)
    return word_list

# tf-idf 获取文本的 top10 关键词
def getKeyWords_tfidf(data, stopkey, topK):
    idList, titleList, abstractList = data['id'], data['title'], data['abstract']
    corpus = []  # 将所有文档输出到一个 list 中，一行就是一个文档
    for index in range(len(idList)):
        text = '%s。%s' % (titleList[index], abstractList) # 拼接标题和摘要
        text = preprocessing(text, stopkey)
        text = " ".join(text)
        corpus.append(text)

    # 1、构建词频矩阵，将文本中的词语转换成词频矩阵
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(corpus) # 词频矩阵，arr[i][j]表示第j个词在第i个文本中的词频
    # 2、统计每个词的tf-idf权值
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(X)
    # 3、获取词袋模型中的关键词
    word = vectorizer.get_feature_names()
    # 4、获取tf-idf矩阵
    weight = tfidf.toarray()
    # 5、打印词语权重
    ids, titles, keys = [], [], []
    for i in range(len(weight)):
        print(u"-------这里输出第", i+1 , u"篇文本的词语tf-idf------")
        ids.append(idList[i])
        titles.append(titleList[i])
        df_word, df_weight = [], []  # 当前文章的所有词汇列表，词汇对应权重列表
        for j in range(len(word)):
            print(word[j], weight[i][j])
            df_word.append(word[j])
            df_weight.append(weight[i][j])
        df_word = pd.DataFrame(df_word, columns=['word'])
        df_weight = pd.DataFrame(df_weight, columns=['weight'])
        word_weight = pd.concat([df_word, df_weight], axis=1) # 拼接词汇列表和权重列表
        word_weight = word_weight.sort_values(by="weight", ascending= False) # 按照权重值降序排列
        keyword = np.array(word_weight['word']) # 选择词汇列并转化为数组格式
        word_split = [keyword[x] for x in range(0, topK)] # 选取前 topK 个作为关键词
        word_split = " ".join(word_split)
        keys.append(word_split)

    result = pd.DataFrame({"id": ids, "title": titles, "key": keys}, columns={'id', 'title', 'key'})
    return  result

def main():
    dataFile = "./data/sample_data.xlsx"  # 获取数据集
    data = pd.read_excel(dataFile)
    stopkey = [word.strip() for word in codecs.open("data/stopWord.txt", "rb").readlines()] # 停用词
    result = getKeyWords_tfidf(data, stopkey, 10)
    result.to_csv("./result/keys_TFIDF.csv", encoding="utf_8_sig", index=False)

if __name__ == '__main__':
    main()















