# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 13:46:08 2017

@author: Administrator
"""

import xlrd
import os
from gensim.models.keyedvectors import KeyedVectors

#from openpyxl import Workbook

#!!!!注意这里的sentence应为列表型
#!!!!这里未考虑两个词和触发词集相似度相同的情况（之后修改可以加上这种情况，若相同或者相差不大，可以根a1文档提供的蛋白质信息得到


def get_trigger(trigger_type_number,prediction_test,test_count):
    trigger_types=[]
    trigger_types.append('Gene_expression')
    trigger_types.append('Transcription')
    trigger_types.append('Protein_catabolism')
    trigger_types.append('Localization')
    trigger_types.append('Binding')
    trigger_types.append('Phosphorylation')
    trigger_types.append('Regulation')
    trigger_types.append('Positive_regulation')
    trigger_types.append('Negative_regulation')
    trigger_types.append('Entity')
    trigger_word='None'
    index=0
    #"F:\\code_project1\\code2.0\\data_09\\training_data\\training_dataClass\\train_trigger.xlsx"
    #"F:\\code_project1\\code2.0\\data_09\\test_data\\training_dataClass\\trainClass_"+str(trigger_type_number)+".xlsx"
    #origin_excel= xlrd.open_workbook("./data_09/training_data/training_dataClass/train_trigger.xlsx")
    #test_excel = xlrd.open_workbook("./data_09/test_data/training_dataClass/trainClass_"+str(trigger_type_number)+".xlsx")
    origin_excel= xlrd.open_workbook("F:\\code_project1\\code2.0\\data_09\\training_data\\training_dataClass\\train_trigger.xlsx")
    test_excel = xlrd.open_workbook("F:\\code_project1\\code2.0\\data_09\\test_data\\new_testdata\\testClassnew_"+str(trigger_type_number)+".xlsx")
    table = origin_excel.sheet_by_name('Sheet')
    table_test = test_excel.sheet_by_name('Sheet')

    name=locals()
    name['class_trigger_type%s' %trigger_type_number] = table.cell(trigger_type_number,2).value.split(',')
    
    #'F:\\code_project1\\code2.0\\model\\300features_40minwords_10context.bin'
    #model = KeyedVectors.load_word2vec_format('./model/300features_40minwords_10context.bin', binary=True)
    model = KeyedVectors.load_word2vec_format('F:\\code_project1\\code2.0\\model\\300features_40minwords_10context.bin', binary=True)
  
          
    k=1
    #初始化标记列表，代表每个句子中还未出现当前类型的触发词
    sentence_flag=[]
    trigger_index=[]
    sentence_before=[]
    for l in range(300000):
        sentence_flag.append(0)
        trigger_index.append([])
        sentence_before.append([])
    trigger_words=[]
    last_sentence_num = 0
    while k<=test_count:
        sentence=table_test.cell(k,1).value 
        txt_num = int(table_test.cell(k,3).value)
        sentence_num = int(table_test.cell(k,2).value)
        if(prediction_test[k-1]==0):
            type_=1
        else:
            type_=0
        #type_代表当前句子中是否存在当前类的触发词
        #若当前句子无所需触发词，则continue读下一个句子
        if(type_ == 0):
            trigger_words.append('NONE')
            k=k+1
            continue
        
        #当前句子存在所需触发词，则先判断是否这个句子在之前已经标记过
        #若已标记则证明该句子存在不止一个触发词，则将其之前的触发词找到，并从句子中删去
        sentence = sentence.split(' ')
        words=[]

        words_=[]
        for wordTag in sentence:
            if(wordTag!=''):
                index = wordTag.index('/')
                word = wordTag[:index]
                words_.append(word)
            else:
                continue
            
        similars=[]
        if(sentence_flag[txt_num*1000+sentence_num]==0):
           sentence_flag[txt_num*1000+sentence_num]=1
           triggerindex_before=-1
        else:            
            triggerindex_before=trigger_index[txt_num*1000+sentence_num][-1]
            sentence=sentence_before[txt_num*1000+sentence_num]
            sentence_flag[txt_num*1000+sentence_num] = sentence_flag[txt_num*1000+sentence_num]+1
            triggerword_before = trigger_word
        #'./a2result/'
        if(os.path.exists('F:\\code_project1\\code2.0\\data_09\\test_data\\new_testdata\\a2result'+str(txt_num)+'.txt')):
            fs = open('F:\\code_project1\\code2.0\\data_09\\test_data\\new_testdata\\a2result'+str(txt_num)+'.txt','a')
        else:
            fs = open('F:\\code_project1\\code2.0\\data_09\\test_data\\new_testdata\\a2result'+str(txt_num)+'.txt','w')
        fs.write(trigger_types[trigger_type_number-1]+',')

        for word_tag in sentence:
            if(word_tag!=''):
                index = word_tag.index('/')
                word = word_tag[:index]
                words.append(word)
                similarity = 0
                i=0
                while i <len(name['class_trigger_type%s' %trigger_type_number]):  
                    if(word not in model.vocab):
                        similarity=0
                        break
                    if(name['class_trigger_type%s' %trigger_type_number][i] not in model.vocab):
                        i=i+1   
                        continue             
                    similarity=similarity+model.similarity(word,name['class_trigger_type%s' %trigger_type_number][i])
                    i=i+1
                similarity=similarity/len(name['class_trigger_type%s' %trigger_type_number])
                if (word in name['class_trigger_type%s' %trigger_type_number] ):
                    similarity = similarity+0.066
                similars.append(similarity)
            else:
                continue
        #！！！！！！！！！！！！这里可以设计一个阈值，因为会出现同一个句子有多个同种触发词的情况
        trigger_word=words[similars.index(max(similars))]#找到当前句的trigger
        index = similars.index(max(similars))  
        index_=words_.index(trigger_word)
        while(index_<index):
            index_=index_+1+words_[index_+1:].index(trigger_word)
        trigger_index[txt_num*1000+sentence_num].append(index) 
        sentence_before[txt_num*1000+sentence_num]=sentence
        sentence_before[txt_num*1000+sentence_num].remove(sentence[similars.index(max(similars))] )   
        fs.write(trigger_word+','+str(sentence_num)+','+str(similars.index(max(similars)))+'\n')#标记当前触发词以及其所在的句子号以及在该句中的第几个单词
        trigger_words.append(trigger_word)#trigger_words是当前测试集所有句子额trigger集合
        k=k+1
        last_sentence_num = sentence_num
        fs.close()
  
    """while j<len(sentence):
        current_word=sentence[j]
        similarity=0
        i=0
        while i<len(name['class_trigger_type%s' %trigger_type_number]):
            similarity=similarity+model.similarity(current_word,name['class_trigger_type%s' %trigger_type_number][i])
            i=i+1
        similarity=similarity/len(name['class_trigger_type%s' %trigger_type_number])
        similars=similars.append(similarity)
        j=j+1
    trigger_words=sentence[similars.index(max(similars))]
    return trigger_words"""
trigger_type_number=6
fss = open("F:\\code_project1\\code2.0\\data_09\\test_data\\new_testdata\\prediction_test_newagain"+str(trigger_type_number)+".txt","r")
l=fss.read()
prediction_test=[]
s=l[1:len(l)-1]
d=s.split(', ')
for i in d:
    prediction_test.append(int(float(i)))
get_trigger(trigger_type_number,prediction_test,2500)
fss.close()