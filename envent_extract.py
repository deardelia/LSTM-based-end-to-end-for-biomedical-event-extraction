# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 19:49:44 2017
第三步：抽取事件（要素提取和事件提取的合并
【用到第二步抽取出来的触发词’T‘开头的信息】
@author: Administrator
"""

import os

#count代表test_data包含的文本个数
count=249
i=43
j=0

#实现根据txt和a1文档，找出每个a1中每个蛋白质所在的句子，同样根据已经提取出的a2部分，找出触发词所在的句子
#在触发词提取阶段，已经找出，每个触发词的类型，触发词本身，以及触发词本身所在的文档位置，存储为了和所给txt同名的a2文件！！！！
while(i<249):   
    if(os.path.exists('F:\\code_project1\\code2.0\\data_09\\test_data\\new5_testdata\\'+'a2result'+str(i)+'.txt')):
        fs_a2=open('F:\\code_project1\\code2.0\\data_09\\test_data\\new5_testdata\\'+'a2result'+str(i)+'.txt','r+')
    else:
        i=i+1
        continue
    fs_a1=open('F:\\code_project1\\code2.0\\data_11\\BioNLP-ST_2011_genia_devel_data_rev1\\deal_data\\'+str(i)+'.a1','r')
    fs_txt=open('F:\\code_project1\\code2.0\\data_11\\BioNLP-ST_2011_genia_devel_data_rev1\\deal_data\\'+str(i)+'.txt','r')
    sentence_end_record=[]
    #!!!!注意这里的原文本有问题（见1.txt）
    sentence=fs_txt.readline()
    k=1
    while sentence!='':
        sentence_end_record.append(fs_txt.tell()-k)
        sentence=fs_txt.readline()
        k=k+1
    sentence_count=len(sentence_end_record)
    k=0
    line=fs_a1.readline()
    a1_count=[]
    a1_record=[]
    #注意a1文件最后的结尾！！！！
    while line!='':
        line=line.split('\t')
        a1_record.append(line)
        num_1=int(line[1].split(' ')[1])
        num_2=int(line[1].split(' ')[2])
        while(num_1>sentence_end_record[k]):
            k=k+1
            
        a1_count.append(k+1)
        line=fs_a1.readline()
    #!!!!注意测试能不能正常写入
    #fs_a1.write(str(a1_count)) 
    #fs_a1.close()
    a2_txt=fs_a2.readlines()
    fs_a2.seek(0)
    line=fs_a2.readline()
    a2_count=[]
    k=0
    event_count=1
    a2_index=0
    while line!='' and line[0]=='T':
        line=line.split(',')
        num_1=int(line[3])
        k=num_1
        num_2=int(line[4].split('\n ')[0])
        """while(num_1>sentence_end_record[k]):
            k=k+1"""
            
        a2_count.append(num_1)
        event=[]
        j=0
        #'Gene_expression'
        if(line[1]=='Gene_expression'):
            event.append('E'+str(event_count))
            event_count=event_count+1
            event.append('Gene_expression:'+line[0])
            while j<len(a1_count):
                if(a1_count[j]==num_1):
                      event.append('Theme:'+a1_record[j][0])
                      break
                else:
                    j=j+1

        #'Transcription'
        if(line[1]=='Transcription'):
            event.append('E'+str(event_count))
            event.append('Transcription:'+line[0])
            j=0
            while j<len(a1_count):
                if(a1_count[j]==num_1):
                      event.append('Theme:'+a1_record[j][0])
                      break
                else:
                    j=j+1                    
        
        #'Protein_catabolism'
        if(line[1]=='Protein_catabolism'):
            event.append('E'+str(event_count))
            event_count=event_count+1
            event.append('Protein_catabolism:'+line[0])
            while j<len(a1_count):
                if(a1_count[j]==num_1):
                      event.append('Theme:'+a1_record[j][0])
                      break
                else:
                    j=j+1      
                    
      #Localization
        if(line[1]=='Localization'):
            event.append('E'+str(event_count))
            event_count=event_count+1
            event.append('Localization:'+line[0])
            while j<len(a1_count):
                if(a1_count[j]==num_1):
                      event.append('Theme:'+a1_record[j][0])
                      break
                else:
                    j=j+1            
            #找atloc或toloc
            if(a2_index==0 and len(a2_txt)>1):
                if(a2_txt[1].split(',')[0][0]=='E' and int(a2_txt[1].split(',')[5])<sentence_end_record[k]):
                    #注意！！！这里只考虑atloc而未考虑toloc
                    event.append('AtLoc:'+a2_txt[1].split(',')[1])
            elif(a2_index!=0):
                if(a2_txt[a2_index-1].split(',')[0][0]=='E'and int(a2_txt[a2_index-1].split(',')[5])<sentence_end_record[k]):
                    #注意！！！这里只考虑atloc而未考虑toloc
                    event.append('AtLoc:'+a2_txt[a2_index-1].split(',')[1])
                elif(a2_index+1<len(a2_txt)):
                    if(a2_index+1<len(a2_txt) and a2_txt[a2_index+1].split(',')[0][0]=='E'and int(a2_txt[a2_index+1].split(',')[5])<sentence_end_record[k]):
                    #注意！！！这里只考虑atloc而未考虑toloc
                        event.append('AtLoc:'+a2_txt[a2_index+1].split(',')[1])
        #Binding
        if(line[1]=='Binding'):
            event.append('E'+str(event_count))
            event_count=event_count+1
            event.append('Binding:'+line[0])
            while j<len(a1_count)-1:
                if(a1_count[j]==k):
                      if(a1_count[j+1]==num_1):
                          event.append('Theme:'+a1_record[j][0]+','+a1_record[j+1][0])
                          j=j+1
                          break
                      else:
                          event.append('Theme:'+a1_record[j][0])
                          break
                else:
                    j=j+1            
            #找site!!!注意：这里只考虑一个site的情况
            if(a2_index==0 and len(a2_txt)>1):
                if(a2_txt[1].split(',')[0][0]=='E' and int(a2_txt[1].split(',')[5])<sentence_end_record[k]):
                    #注意！！！这里只考虑atloc而未考虑toloc
                    event.append('Site:'+a2_txt[1].split(',')[1])
            elif(a2_index!=0):
                if(a2_txt[a2_index-1].split(',')[0][0]=='E'and int(a2_txt[a2_index-1].split(',')[5])<sentence_end_record[k]):
                    #注意！！！这里只考虑atloc而未考虑toloc
                    event.append('Site:'+a2_txt[a2_index-1].split(',')[1])
                elif(a2_index+1<len(a2_txt)):
                    if(a2_txt[a2_index+1].split(',')[0][0]=='E'and int(a2_txt[a2_index+1].split(',')[5])<sentence_end_record[k]):
                    #注意！！！这里只考虑atloc而未考虑toloc
                        event.append('Site:'+a2_txt[a2_index+1].split(',')[1])
                    
            #Phosphorylation
        if(line[1]=='Phosphorylation'):
            event.append('E'+str(event_count))
            event_count=event_count+1
            event.append('Phosphorylation:'+line[0])
            while j<len(a1_count):
                if(a1_count[j]==num_1):
                      event.append('Theme:'+a1_record[j][0])
                      break
                else:
                    j=j+1            
            #找site
            if(a2_index==0 and len(a2_txt)>1):
                if(a2_txt[1].split(',')[0][0]=='E' and int(a2_txt[1].split(',')[5])<sentence_end_record[k]):
                    #注意！！！这里只考虑atloc而未考虑toloc
                    event.append('Site:'+a2_txt[1].split(',')[1])
            elif(a2_index!=0):
                if(a2_txt[a2_index-1].split(',')[0][0]=='E'and int(a2_txt[a2_index-1].split(',')[5])<sentence_end_record[k]):
                    #注意！！！这里只考虑atloc而未考虑toloc
                    event.append('Site:'+a2_txt[a2_index-1].split(',')[1])
                elif(a2_index+1<len(a2_txt)):
                    
                    if( a2_txt[a2_index+1].split(',')[0][0]=='E'and int(a2_txt[a2_index+1].split(',')[5])<sentence_end_record[k]):
                    #注意！！！这里只考虑atloc而未考虑toloc
                        event.append('Site:'+a2_txt[a2_index+1].split(',')[1])
        #这六种分析完后就写进a2（单独分析剩下的三种，因为剩下的三种需要这六类event的信息）
        m=0
        while m<len(event):
            if(m==0):
                fs_a2.write(event[m])
                fs_a2.write('\t')
            else:
                fs_a2.write(event[m]+' ')
            m=m+1
        fs_a2.write('\n')
        line=fs_a2.readline()
        a2_index=a2_index+1
    i=i+1
    #fs_a1.close()
    fs_txt.close()
    fs_a1.close()
    fs_a2.close()
    