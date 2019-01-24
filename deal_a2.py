# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 19:25:13 2017

@author: Administrator
"""
import os
"""
i=1
count=249

while i<=count:
    if(os.path.exists('F:\\code_project1\\code2.0\\data_09\\test_data\\new_testdata\\'+'a2result'+str(i)+'.txt')):
        fs_a2=open('F:\\code_project1\\code2.0\\data_09\\test_data\\new_testdata\\'+'a2result'+str(i)+'.txt','r')
        fs_a1=open('F:\\code_project1\\code2.0\\data_11\\BioNLP-ST_2011_genia_devel_data_rev1\\deal_data\\'+str(i)+'.a1','r')
    else:
        i=i+1
        continue
    a1=fs_a1.readlines()
    numlable=int(a1[-1].split('\t')[0][1:])
    a2=fs_a2.readlines()
    fs_newa2=open('F:\\code_project1\\code2.0\\data_09\\test_data\\new2_testdata\\'+'a2result'+str(i)+'.txt','w')
    for sentence in a2:
        if(sentence=='' or sentence=='\n'):
            continue
        numlable=numlable+1
        fs_newa2.write('T'+str(numlable)+',')
        fs_newa2.write(sentence)
    fs_newa2.close()
    fs_a2.close()
    fs_a1.close()
    i=i+1
    
#排序
i=1
count=259
while i<=count:
    if(os.path.exists('F:\\code_project1\\code2.0\\data_09\\test_data\\new2_testdata\\'+'a2result'+str(i)+'.txt')):
        fs1=open('F:\\code_project1\\code2.0\\data_09\\test_data\\new2_testdata\\'+'a2result'+str(i)+'.txt','r')
        fs2=open('F:\\code_project1\\code2.0\\data_09\\test_data\\new3_testdata\\'+'a2result'+str(i)+'.txt','w')
    else:
        i=i+1
        continue
    sort1=[]
    sort2=[]
    index=[]
    text=fs1.readlines()
    #记录下每个a2文档中每条记录的所在句子以及在该句子中的位置,index用来记录重排后的记录号
    for sentence in text:
        s=sentence.split(',')
        sort1.append(int(s[3]))
        sort2.append(int(s[4].split('\n')[0]))
    sorts=[]
    for x in sort1:
        sorts.append(x)
    n=0
    while n<len(sort2):
        maxi=sort1.index(min(sort1))
        index.append(maxi)
        sort1[maxi]=10000+n
        n=n+1
    fs1.seek(0)
    text=fs1.readlines()
    text_tmp=[]
    sortindex=[]
    sortsentence=[]
    for n in index:
        text_tmp.append(text[n])
        sortindex.append(sort2[n])
        sortsentence.append(sorts[n])
        
    index=[]
    n=0
    while n<len(sortsentence)-1:
        tmp=[]
        index.append(n)
        tmp.append(sortindex[n])
        while n<len(sortsentence)-1:
            if(sortsentence[n]==sortsentence[n+1]):
                tmp.append(sortindex[n+1])
                n=n+1
            else:
                break
        if(len(tmp)==1):
            n=n+1
            continue
        else:
            index.remove(index[len(index)-1])
            kk=[]
            for ll in tmp:
                kk.append(ll)
            ll=0
            while ll < len(kk):
                index_=tmp.index(min(tmp))
                tmp[index_]=10000+ll
                index.append(n-len(kk)+1+index_)
                ll=ll+1
            n=n+1
    if(len(index)<len(text_tmp)):
        index.append(len(text_tmp)-1)
    text_end=[]
    for m in index:
        text_end.append(text_tmp[m])
    for sentence in text_end:
        fs2.write(sentence)
        
    fs1.close()
    fs2.close()
    i=i+1
i=1
count=259

while i<=count:
    if(os.path.exists('F:\\code_project1\\code2.0\\data_09\\test_data\\new3_testdata\\'+'a2result'+str(i)+'.txt')):
        fs1=open('F:\\code_project1\\code2.0\\data_09\\test_data\\new3_testdata\\'+'a2result'+str(i)+'.txt','r')
        fs2=open('F:\\code_project1\\code2.0\\data_09\\test_data\\new4_testdata\\'+'a2result'+str(i)+'.txt','w')
        fs_a1=open('F:\\code_project1\\code2.0\\data_11\\BioNLP-ST_2011_genia_devel_data_rev1\\deal_data\\'+str(i)+'.a1','r')
    else:
        i=i+1
        continue
    k=0
    text=fs1.readlines()
    a1=fs_a1.readlines()
    numlable=int(a1[-1].split('\t')[0][1:])+1
    while k< len(text):
        sentence=text[k]
        fs2.write('T'+str(numlable))
        for word in sentence.split(',')[1:]:
            fs2.write(','+word)
        numlable=numlable+1
        k=k+1
    i=i+1
    fs1.close()
    fs2.close()
    fs_a1.close()

"""
i=1
count=249
while i< count:
    if(os.path.exists('F:\\code_project1\\code2.0\\data_09\\test_data\\new4_testdata\\'+'a2result'+str(i)+'.txt')):

        fs1=open('F:\\code_project1\\code2.0\\data_09\\test_data\\new4_testdata\\'+'a2result'+str(i)+'.txt','r')
        fs2=open('F:\\code_project1\\code2.0\\data_09\\test_data\\new5_testdata\\'+'a2result'+str(i)+'.txt','w')
        fs_txt=open('F:\\code_project1\\code2.0\\data_11\\BioNLP-ST_2011_genia_devel_data_rev1\\deal_data\\'+str(i)+'.txt','r')
    else:
        i=i+1
        continue
    record=[]
    f1=fs1.readlines()
    sentence=fs_txt.readline()
    k=1
    while sentence!='':
        record.append(fs_txt.tell()-k)
        sentence=fs_txt.readline()
        k=k+1
    fs_txt.seek(0)
    sentences=fs_txt.readlines()
    for a2 in f1:
        a2=a2.split(',')
        begin=record[int(a2[3])-2]+sentences[int(a2[3])-1].find(a2[2])+1
        end=begin+len(a2[2])
        for key in a2:
            if(key[-1]=='\n'):
                fs2.write(key[0:len(key)-1])
                fs2.write(',')
            else:
                fs2.write(key)
                fs2.write(',')
        fs2.write(str(begin))
        fs2.write(',')
        fs2.write(str(end)+'\n')
    
    fs2.close()
    fs1.close()
    fs_txt.close()
    i=i+1