# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 10:25:36 2021

@author: Administrator
"""

import jieba

sentence = "第二次来吃了，味道还可以，就在金港国际里面，逛累的可以来吃"

seg_lst1 = jieba.cut(sentence, cut_all=True)
seg_lst2 = jieba.cut(sentence, cut_all=False)

print("全模式  :" + "/".join(seg_lst1))
print("精确模式:" + "/".join(seg_lst2))