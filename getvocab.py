# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 22:21:38 2017

@author: Administrator
"""
import gensim  
model = gensim.models.KeyedVectors.load_word2vec_format("F:\\pubmed_bin\\wikipedia-pubmed-and-PMC-w2v.bin",binary=True) 
vocab=model.vocab
i=0
for m in vocab:
    if(i>10):
        break
    i=i+1
    print(m)
