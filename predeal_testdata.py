# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 21:48:06 2017

@author: Administrator
"""

count=1
max_count=259
while count <=max_count:
    fs=open('F:\\code_project1\\code2.0\\data_11\\BioNLP-ST_2011_genia_devel_data_rev1\\BioNLP-ST_2011_genia_devel_data_rev1\\'+'1 '+'('+str(count)+')'+'.txt','r')
    fs_out=open('F:\\code_project1\\code2.0\\data_11\\BioNLP-ST_2011_genia_devel_data_rev1\\deal_data\\'+str(count)+'.txt','w')
    text=fs.readlines()
    if(text==''):
        fs_out.write('')
        fs.close()
        fs_out.close()
        count=count+1
        continue
    if(len(text)==1):
        if('. ' in text[0]):
            current_txt=text[0].split('. ')
            for sen in current_txt:
                fs_out.write(sen)
                fs_out.write('.\n')
            fs_out.close()
            count=count+1
            fs.close()
            continue
        else:
            fs_out.write(text[0])
            fs.close()
            fs_out.close()
            count=count+1
            continue
        
    fs_out.write(text[0])
    if(len(text)>=2):
        for k in range(len(text)):
            l=0
            if k==0:
                continue
            while l < len(text[k].split('. ')):
                if(text[k].split('. ')[l]=='' or text[k].split('. ')[l]=='\n'):
                    l=l+1
                    continue
                fs_out.write(text[k].split('. ')[l])
                #fs_out.write('\n')
                if(l+1==len(text[k].split('. '))):
                    l=l+1
                    continue
                fs_out.write('.\n')
                l=l+1
        fs.close()
        fs_out.close()
        count=count+1
        continue
    split_sentence=text[1].split('. ')
    i=0
    while i<len(split_sentence):
        fs_out.write(split_sentence[i])
        fs_out.write('.\n')
        i=i+1
    fs.close()
    fs_out.close()
    count=count+1
    
"""    
count=1
max_count=259
while count <=max_count:
    fs=open('F:\\code_project1\\code2.0\\data_11\\BioNLP-ST_2011_genia_devel_data_rev1\\BioNLP-ST_2011_genia_devel_data_rev1\\'+'1 '+'('+str(count)+')'+'.a1','r')
    fs_out=open('F:\\code_project1\\code2.0\\data_11\\BioNLP-ST_2011_genia_devel_data_rev1\\deal_data\\'+str(count)+'.a1','w')
    text=fs.readlines()
    i=0
    while i<len(text):
        fs_out.write(text[i])
        i=i+1
    fs.close()
    fs_out.close()
    count=count+1
"""