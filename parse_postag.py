# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 20:40:19 2017
第一步
数据来源：来自PubMed的数据
目标：进行分词分句
问题：目前生成的是csv文件（效果如何还需测试
@author: Administrator
"""
from nltk import word_tokenize, pos_tag
import csv

count=1
max_count=5000000

result=[]

new_sentence_file=open('F:\\PubMedSpider\\new_senenceCSV.csv','r+')
writeCSV=csv.writer(new_sentence_file)

while count<max_count:
    fp=open('F:\\PubMedSpider\\together.txt','r')
    #fp2=open('F:\\PubMedSpider\\new_together.txt','r+')#一定要先建立一个空的才可以用r+
    sentences=fp.readlines()
    for sentence in sentences:
        if (sentence=='[Title]\n') or (sentence == '[Astract]\n') or (sentence=='\n'):
            continue
        else:
            sentence = sentence.replace(',','')
            sentence = sentence.replace('.','')
            sentence = sentence.replace('(','')
            sentence = sentence.replace(')','')
            sentence = sentence.replace('[','')
            sentence = sentence.replace(']','')
            sentence = sentence.replace('\"','')
            sentence = sentence.replace(':','')
            sentence = sentence.replace('\'','')
            pos_deal=pos_tag(word_tokenize(sentence))
            new_sentence=pos_deal
            '''new_sentence=''
            for i in pos_deal:
                new_sentence=new_sentence+i[0]+'/'+i[1]+' '
            new_sentence=new_sentence+'\n'
            '''
            #fp2.write(new_sentence)
            writeCSV.writerows(new_sentence)

    fp.close()


