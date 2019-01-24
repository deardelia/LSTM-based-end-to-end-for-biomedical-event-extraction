# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 21:40:27 2017

@author: Administrator
"""

from nltk import word_tokenize, pos_tag

count=1
max_count=800
while count<max_count:
    fs=open('F:\\code_project1\\code2.0\\data_09\\training_data\\'+str(count)+'.txt','r')
    sentences=fs.readlines()
    for sentence in sentences:
        sentence = sentence.replace(',','')
        sentence = sentence.replace('.','')
        sentence = sentence.replace('(','')
        sentence = sentence.replace(')','')
        sentence = sentence.replace('[','')
        sentence = sentence.replace(']','')
        sentence = sentence.replace('\"','')
        sentence = sentence.replace(':','')
        sentence = sentence.replace('\'','')
        sentence = sentence.replace('/',' ')
        sentence = sentence.replace('\n','')
        pos_deal = pos_tag(word_tokenize(sentence))
    count=count+1
    fs.close()