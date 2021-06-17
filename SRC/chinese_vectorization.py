# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 10:01:45 2021

@author: Administrator

使用Word2Vec进行中文向量化
"""

import gensim.models


my_corpus = [
    ['凉皮', '有', '味道', '了', '吃完', '一天', '肚子', '都', '不舒服'],
    ['帅哥', '经理', '又', '帅', '服务', '又', '好', '凉皮', '味道', '又', '不', '吃完', '肚子', '撑']
]

"""
由于语料库过小，因此设置min_count为2，embedding在10维的向量空间
"""
model = gensim.models.Word2Vec(sentences=my_corpus, min_count=2, vector_size=10)

"""
也可以分步进行模型的训练，这样对于某些不可重复的流（Kafka）可以手动控制
model = gensim.models.Word2Vec(params) # an empty model, not taining yet
model.build_vocab(sentences) 
mode.train(other_sentences)
"""

for index, word in enumerate(model.wv.index_to_key):
    print("Index #[{}] word: {}".format(index, word))


for v in range(len(model.wv)):
    print(model.wv[v])