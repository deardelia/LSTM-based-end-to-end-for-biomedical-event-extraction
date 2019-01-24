# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 23:52:13 2017

@author: Administrator
"""

fs=open('F:\\code_project1\\code2.0\\argument_detect\\train_relations.txt','r')
fs1=open('F:\\code_project1\\code2.0\\argument_detect\\train_relationsv3.txt','w')

i=1
text=fs.readlines()
for m in text:
    fs1.write(str(i)+'\t'+m)
    i=i+1
    