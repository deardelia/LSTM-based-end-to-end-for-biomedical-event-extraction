# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 21:04:50 2017

@author: Administrator
"""

count=1
max_count=1797
while count <=max_count:
    fs=open('F:\\code_project1\\traning_data\\TRAIN\TRAIN\\'+'1 '+'('+str(count)+')'+'.txt','r')
    fs_out=open('F:\\code_project1\\code2.0\\data_09\\training_data\\'+str(count)+'.txt','w')
    text=fs.readlines()
    if(text==''):
        fs_out.write('')
        fs.close()
        fs_out.close()
        count=count+1
        continue
    if(len(text)==1):
        fs_out.write(text[0])
        fs.close()
        fs_out.close()
        count=count+1
        continue
    fs_out.write(text[0])
    split_sentence=text[1].split('. ')
    i=0
    while i<len(split_sentence):
        fs_out.write(split_sentence[i])
        fs_out.write('.\n')
        i=i+1
    fs.close()
    fs_out.close()
    count=count+1"""
    
count=1
max_count=1797
while count <=max_count:
    fs=open('F:\\code_project1\\traning_data\\TRAIN\TRAIN\\'+'1 '+'('+str(count)+')'+'.a1','r')
    fs_out=open('F:\\code_project1\\code2.0\\data_09\\training_data\\'+str(count)+'.a1','w')
    text=fs.readlines()
    i=0
    while i<len(text):
        fs_out.write(text[i])
        i=i+1
    fs.close()
    fs_out.close()
    count=count+1