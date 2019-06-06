# coding=utf-8
# 采用Word2Vec词聚类方法抽取关键词2——根据候选关键词的词向量进行聚类分析

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import math
import os

# 对词向量采用 K-means 聚类抽取 TopK 关键词
def getKeywords_kmeans(data, topK):
    words = data['word'] # 词汇
    vectors = data.ix[:, 1:] # 拿到向量

    kmeans = KMeans(n_clusters=1, random_state=10).fit(vectors)
    labels = kmeans.labels_    # 类别结果标签
    labels = pd.DataFrame(labels, columns=['label'])
    new_df = pd.concat([labels, vectors], axis=1)
    df_count_type = new_df.groupby('label').size()  # 各类别统计个数
    # print(df_count_type)
    vector_center = kmeans.cluster_centers_  # 聚类中心

    # 计算距离（相似性） 采用欧式距离
    distances = []
    vector_words = np.array(vectors) # 候选关键词向量，转化为 array
    vector_center = vector_center[0] # 第一个类别聚类中心，本例只有一个类别
    length = len(vector_center)  # 向量的维度
    for index in range(len(vector_words)): # 遍历候选关键词
        cur_word_vec = vector_words[index] # 当前词语的词向量
        dis = 0  # 向量距离
        for index2 in range(length):
            dis = dis + (vector_center[index2] - cur_word_vec[index2]) * (vector_center[index2] - cur_word_vec[index2])
        dis = math.sqrt(dis)
        distances.append(dis)
    distances = pd.DataFrame(distances, columns=['dis'])

    result = pd.concat([words, labels, distances], axis=1) # 拼接词语与其对应中心点的距离
    result = result.sort_values(by="dis", ascending=True)

    # 抽取排名前 topk 个词语作为文本关键词
    wordList = np.array(result['word'])  # 选择词汇
    word_split = [wordList[x] for x in range(0, topK)] # 抽取前 topK 个词汇
    word_split = " ".join(word_split)
    return word_split

def main():
    dataFile = "data/sample_data.xlsx"
    raw_data = pd.read_excel(dataFile)
    ids, titles, keys = [], [], []

    rootdir = "result/vectors"  # 词向量根目录文件
    fileList = os.listdir(rootdir)  # 该目录下所有的文件
    for i in range(len(fileList)): # 遍历所有的文件
        filename = fileList[i]
        path = os.path.join(rootdir, filename)
        if os.path.isfile(path):
            data = pd.read_csv(path, encoding='utf-8') # 读取词向量文件
            article_keys = getKeywords_kmeans(data, 10) # 聚类算法得到当前文件的关键词
            # 根据文件名获取文章的 id 和 标题
            (shortname, extension) = os.path.split(filename) # 得到文件名和文件扩招名
            t = shortname.split("_")
            article_id = int(t[len(t) - 1]) # 获取文章的 id
            article_title = data[data.id == article_id]['title']  # 获得文章标题
            article_title = list(article_title)[0]  # 转化为 list
            ids.append(article_id)
            titles.append(article_title)
            keys.append(article_keys.encode("utf-8"))

    # 所有结果写入文件
    result = pd.DataFrame({"id": ids, "title": titles, "key": keys}, columns=['id', 'title', 'key'])
    result = result.sort_values(by="id",ascending=True)   # 排序
    result.to_csv("result/keys_word2vec.csv", index=False)

if __name__ == '__main__':
    main()









