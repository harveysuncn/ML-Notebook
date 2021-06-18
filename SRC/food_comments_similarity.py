# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 13:24:01 2021

@author: Administrator

使用O2O商铺食品安全相关评论生成Wrod2Vec模型
并对词向量进行相似度的计算
"""

import pandas
import jieba
import gensim.models

class Comments:
    ''' Load Comments '''
    def __iter__(self):
        # comments corpus 
        file_path = "..\\Datasets\\datafountain_O2O_comments\\train.csv"
        comments = pandas.read_csv(file_path, sep="\t")['comment']
        for comment in comments:
            yield comment
            

    
class CommentCorpus:
    ''' Comment Corpus '''
    def __init__(self):
        self.raw_data = Comments()
        # stopwords corpus
        stop_words_path = "..\\Datasets\\cn_stopwords.txt"            
        self.stop_words = [line.strip() for line in 
                           open(stop_words_path, encoding="utf-8").readlines()]
        
    def __iter__(self):
        for comment in self.raw_data:
            word_lst = []
            cut_lst = jieba.lcut(comment, cut_all=False)
            for word in cut_lst:
                if word not in self.stop_words:
                    word_lst.append(word)
            yield word_lst


def demo(model):
    print("Words which similar to '好吃':")
    output1 = model.wv.most_similar(positive=['好吃'])
    print(output1)
    
    print("Words which contrast to '难吃':")
    output2 = model.wv.most_similar(negative=['难吃'])
    print(output2)
    
    print("气质不合的词语：")
    output3 = model.wv.doesnt_match(
        ['好吃', '难吃', '蟑螂', '老鼠', '恶心'])
    print(output3)
    
    print("相似度查询：")
    queries = (
        ['好吃', '推荐'],
        ['蟑螂', '难吃'],
        ['辣',   '难吃']
        )
    for query in queries:
        print(model.wv.similarity(*query))
    
    


if __name__ == '__main__':
    com_corpus = CommentCorpus()
    model = gensim.models.Word2Vec(com_corpus, min_count=5)
    
    demo(model)
