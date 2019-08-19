# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 17:04:30 2017

@author: Administrator
"""
import nltk
i=1
count=800

def get_proteinnumber(cur,E_store,T_store):
    if(cur[1][-1]=='\n'):
        num=int(cur[1][1:-1])
    else:
        num=int(cur[1][1:])
        
    if(cur[1][:1]=='T'):
        if(num<T_fenjie):
            num1_p=int(T_store[num-1].split('\t')[1].split(' ')[1])
            num2_p=int(T_store[num-1].split('\t')[1].split(' ')[2])
            p_name=T_store[num-1].split('\t')[2][:-1]
        else:
            num1_p=int(T_store[num-1].split('\t')[1].split(' ')[1])
            num2_p=int(T_store[num-1].split('\t')[1].split(' ')[2])
            p_name=T_store[num-1].split('\t')[2][:-1]
    else:
        event=E_store[num-1]
        event=event.split('\t')[1].split(' ')
        cur=event[1].split(':')  
        num1_p,num2_p,p_name=get_proteinnumber(cur,E_store,T_store)
        
    return num1_p,num2_p,p_name

                        
def get_otherS(curen_sentence,sentences,curen_e2,curen_e1):
    
    pos_sentence=nltk.word_tokenize(sentences[curen_sentence])
    token_sentence=nltk.word_tokenize(sentences[curen_sentence])
    pos_sentence=nltk.pos_tag(pos_sentence)
    other_sentences=[]
    i=0
    place_e1=0
    for m in token_sentence:
        if(m==curen_e1[0]):
            place_e1=i
            break
        else:
            i=i+1
    j=0
    for n in pos_sentence:
        if((n[1]=='VB'or n[1]=='VBP'or n[1]=='VBD'or n[1]=='VBG' or n[1]=='VBZ' )and n[0] not  in curen_e2 and n[0] not in curen_e1):
            if(k>place_e1):
                sentence=' '.join(token_sentence[:place_e1])+' '+'<e1>'+curen_e1[0]+'</e1>'+' '+' '.join(token_sentence[place_e1+1:j])+' '+'<e2>'+token_sentence[j]+'</e2>'+' '.join(token_sentence[j+1:])
                other_sentences.append(sentence)
                j=j+1
            else:
                sentence=' '.join(token_sentence[:j])+' '+'<e1>'+token_sentence[j]+'</e1>'+' '+' '.join(token_sentence[j+1:place_e1])+' '+'<e2>'+token_sentence[place_e1]+'</e2>'+' '.join(token_sentence[place_e1+1:])
                other_sentences.append(sentence)
                j=j+1
        else:
            j=j+1
    return other_sentences
while i<=count:
    fs=open('F:\\code_project1\\code2.0\\data_09\\bionlp09_shared_task_training_data_rev2\\1 ('+str(i)+').txt','r')
    fs_=open('F:\\code_project1\\code2.0\\data_09\\bionlp09_shared_task_training_data_rev2\\deal\\'+str(i)+'.txt','w')
    old=fs.readlines()
    for m in old:
        fs_.write(m[:-1]+' ')
    fs.close()
    fs_.close()
    i=i+1

i=1
while i<=count:
    fs1=open('F:\\code_project1\\code2.0\\data_09\\bionlp09_shared_task_training_data_rev2\\deal\\'+str(i)+'.txt','r')
    fs_a1=open('F:\\code_project1\\code2.0\\data_09\\bionlp09_shared_task_training_data_rev2\\1 ('+str(i)+').a1','r')
    fs_a2=open('F:\\code_project1\\code2.0\\data_09\\bionlp09_shared_task_training_data_rev2\\1 ('+str(i)+').a2','r')
    fs_new=open('F:\\code_project1\\code2.0\\data_09\\bionlp09_shared_task_training_data_rev2\\new\\'+str(i)+'.txt','w')
    text=fs1.read()
    tmp_sentence=0
    curen_sentence=0
    sentences=text.split('. ')
    curen_e2=[]
    curen_e1=[]
    send_rec=[]
    length=0
    record=0
    for m in sentences:
        length=length+len(m)+2
        send_rec.append(length)
    a1=fs_a1.readlines()
    a2=fs_a2.readlines()
    if(a2==[] or a1==[]):
        i=i+1
        continue
    k=0
    kk=0
    T_store=[]
    E_store=[]
    for m in a1:
        T_store.append(m)
    T_fenjie=len(T_store)
    while(k<len(a2)):
        if(a2[k].split('\t')[0][0]=='E'):
            E_store.append(a2[k])
            k=k+1
            continue
        else:
            k=k+1
            continue
    k=0   
    while( k<len(a2) and a2[k].split('\t')[0][0]!='E'):
        if(a2[k].split('\t')[0][0]=='T'):
            T_store.append(a2[k])
        k=k+1
    k=0
    theme_num1=0
    theme_num2=0
    while(k<len(a2)):
        if(a2[k].split('\t')[0][0]!='E'):
            k=k+1
            continue
        else:
            kk=0
            event=a2[k].split('\t')[1].split(' ')
            num_e=int(event[0].split(':')[1][1:])
            num1_e=int(T_store[num_e-1].split('\t')[1].split(' ')[1])
            num2_e=int(T_store[num_e-1].split('\t')[1].split(' ')[2])
            curen_e2.append(text[num1_e:num2_e])
            while kk<len(event)-1:
                if(event[kk+1]==''):
                    kk=kk+1
                    continue
                cur=event[kk+1].split(':')
                num1_p,num2_p,p_name=get_proteinnumber(cur,E_store,T_store)
                curen_e1.append(text[num1_p:num2_p])
                num1_etmp =num1_e+1
                num2_etmp =num2_e+1
                
                if(event[kk+1].split(':')[0]=='Theme'):
                    theme_num1=num1_p+1
                    theme_num2=num2_p+1
                    
                if(event[kk+1].split(':')[0]=='Site'):
                    num1_e=theme_num1-1
                    num2_e=theme_num2-1
                rel=cur[0]
                max_num=max(num1_p,num2_p,num1_e,num2_e)
                curen_s=0
                while(curen_s<len(send_rec)):
                    if(send_rec[curen_s]>max_num):
                        break
                    else:
                        curen_s=curen_s+1

                if(curen_s>0):
                    #if(kk==0):
                    num1_e=num1_e-send_rec[curen_s-1]-1
                    num2_e=num2_e-send_rec[curen_s-1]
                    if(num1_e==-1):
                        num1_e=0
                    """if(event[kk+1].split(':')[0]=='Site'):
                        num1_e=num1_e-send_rec[curen_s-1]-1
                        num2_e=num2_e-send_rec[curen_s-1]"""
                    num1_p=num1_p-send_rec[curen_s-1]-1
                    num2_p=num2_p-send_rec[curen_s-1]
                    if(num1_p==-1):
                        num1_p=0

                if(num1_e<num1_p):
                    fs_new.write(sentences[curen_s][:num1_e]+'<e1>'+sentences[curen_s][num1_e:num2_e]+'</e1>')
                    fs_new.write(sentences[curen_s][num2_e:num1_p]+'<e2>'+sentences[curen_s][num1_p:num2_p]+'</e2>'+sentences[curen_s][num2_p:]+'.\n')
                else:
                    if(num1_p>=0):
                        fs_new.write(sentences[curen_s][:num1_p]+'<e1>'+sentences[curen_s][num1_p:num2_p]+'</e1>')
                        fs_new.write(sentences[curen_s][num2_p:num1_e]+'<e2>'+sentences[curen_s][num1_e:num2_e]+'</e2>'+sentences[curen_s][num2_e:]+'.\n')
                    else:
                        fs_new.write('<e1>'+p_name+'</e1>'+sentences[curen_s][:num1_e]+'<e2>'+sentences[curen_s][num1_e:num2_e]+'</e2>'+sentences[curen_s][num2_e:]+'.\n')
                fs_new.write(rel+'(e1,e2)\n')
                num1_e=num1_etmp-1
                num2_e=num2_etmp-1
                kk=kk+1
            tmp_sentence=curen_s
            if(tmp_sentence!=curen_sentence):
                other_sentences=get_otherS(curen_sentence,sentences,curen_e2,curen_e1)
                curen_e2=[]
                curen_e1=[]
                for Osentence in other_sentences:
                    fs_new.write(Osentence+'\n')
                    fs_new.write('Other\n')
                curen_sentence=tmp_sentence
            k=k+1
    fs_new.close()
    fs_a2.close()
    fs_a1.close()
    fs1.close()
    i=i+1       

i=1
count=800
num=1
n=1
while i<=count:
    num=1
    fs1=open('F:\\code_project1\\code2.0\\data_09\\bionlp09_shared_task_training_data_rev2\\new\\'+str(i)+'.txt','r')
    fs2=open('F:\\code_project1\\code2.0\\data_09\\bionlp09_shared_task_training_data_rev2\\end\\'+str(i)+'.txt','w')
    s=fs1.readline()
    while(s!='' and s!='\n'):    
        fs2.write(str(num)+'\t'+'\"'+s[:-1]+'\"'+'\n')
        s=fs1.readline()
        if(s.split('(')[0]=='Theme1'or s.split('(')[0]=='Theme2' or s.split('(')[0]=='Theme3' or s.split('(')[0]=='Theme4' or s.split('(')[0]=='Theme5'):
            fs2.write('Theme'+'('+s.split('(')[1])
        elif(s.split('(')[0]=='Site1' or s.split('(')[0]=='Site2' or s.split('(')[0]=='Site3' or s.split('(')[0]=='Site4' or s.split('(')[0]=='Site5' ):
            fs2.write('Site'+'('+s.split('(')[1])
        else:
            fs2.write(s)
        fs2.write('Comment:\n')
        fs2.write('\n')
        s=fs1.readline()
        num=num+1
    fs1.close()
    fs2.close()   
    i=i+1
            
n=1
count=800
num=1
fs2=open('F:\\code_project1\\code2.0\\data_09\\bionlp09_shared_task_training_data_rev2\\end\\'+'final'+'.txt','w')

while n<=count:
    fs1=open('F:\\code_project1\\code2.0\\data_09\\bionlp09_shared_task_training_data_rev2\\end\\'+str(n)+'.txt','r')
    text_curen=fs1.readlines()
    if(text_curen==[]):
        n=n+1
        continue
    i=1
    fs2.write(str(num)+'\t'+text_curen[0].split('\t')[1])
    fs2.write(text_curen[i])
    i=i+1
    fs2.write(text_curen[i])
    i=i+1
    fs2.write(text_curen[i])
    i=i+1
    num=num+1
    
    while i<len(text_curen):
        if(text_curen[i-1]=='\n'):
            fs2.write(str(num)+'\t'+text_curen[i].split('\t')[1])
            num=num+1
            i=i+1
            fs2.write(text_curen[i])
            i=i+1
            fs2.write(text_curen[i])
            i=i+1
            fs2.write(text_curen[i])
            i=i+1
    fs1.close()
    n=n+1
fs2.close()
   
i=0 
fs=open('F:\\code_project1\\code2.0\\data_09\\bionlp09_shared_task_training_data_rev2\\end\\'+'final'+'.txt','r')
fs1=open('F:\\code_project1\\code2.0\\data_09\\bionlp09_shared_task_training_data_rev2\\end\\'+'final_end'+'.txt','w')
text=fs.readlines()
flag=0
flag2=0
while i <len(text):
    if(i%4==0):
        e1=text[i].find('<e1>')
        e2=text[i].find('<e2>')
        if(e1<e2):
            s=text[i].split('<e1>')
            event=s[1].split('</e1>')
            if(event[0][0]==' ' or event[0][-1]==' '):
                if(event[0][0]==' ' and event[0][-1]!=' '):
                    fs1.write(s[0]+' '+'<e1>')
                    fs1.write(event[0][1:]+'</e1>')
                elif(event[0][0]==' ' and event[0][-1]==' '):
                    flag=1
                    fs1.write(s[0]+' '+'<e1>')
                    fs1.write(event[0][1:-1]+'</e1>')
                else:
                    flag=1
                    fs1.write(s[0]+'<e1>')
                    fs1.write(event[0][:-1]+'</e1>')
            else:
                fs1.write(s[0]+'<e1>')
                fs1.write(event[0]+'</e1>')
            e2s=event[1].split('<e2>')
            if(flag==1):
                fs1.write(' '+e2s[0])
                flag=0
            else:
                fs1.write(e2s[0])
            event2=e2s[1].split('</e2>')
            if(event2[0][0]==' ' or event2[0][-1]==' '):
                if(event2[0][0]==' ' and event2[0][-1]!=' '):
                    fs1.write(' <e2>'+event2[0][1:]+'</e2>')
                elif(event2[0][0]==' ' and event2[0][-1]==' '):
                    flag2=1
                    fs1.write(' <e2>'+event2[0][1:-1]+'</e2>')
                else:
                    flag2=1
                    fs1.write('<e2>'+event2[0][:-1]+'</e2>')
            else:
                fs1.write('<e2>')
                fs1.write(event2[0]+'</e2>')
            if(flag2==1):
                fs1.write(' '+event2[1])
                flag2=0
            else:
                fs1.write(event2[1])

        else:
            s=text[i].split('<e2>')
            event=s[1].split('</e2>')
            if(event[0][0]==' ' or event[0][-1]==' '):
                if(event[0][0]==' ' and event[0][-1]!=' '):
                    fs1.write(s[0]+' '+'<e2>')
                    fs1.write(event[0][1:]+'</e2>')
                elif(event[0][0]==' ' and event[0][-1]==' '):
                    flag=1
                    fs1.write(s[0]+' '+'<e2>')
                    fs1.write(event[0][1:-1]+'</e2>')
                else:
                    flag=1
                    fs1.write(s[0]+'<e2>')
                    fs1.write(event[0][:-1]+'</e2>')
            else:
                fs1.write(s[0]+'<e2>')
                fs1.write(event[0]+'</e2>')
            e2s=event[1].split('<e1>')
            if(flag==1):
                fs1.write(' '+e2s[0]+'<e1>')
                flag=0
            else:
                fs1.write(e2s[0]+'<e1>')
            event2=e2s[1].split('</e1>')
            if(event2[0][0]==' ' or event[0][-1]==' '):
                if(event2[0][0]==' ' and event2[0][-1]!=' '):
                    fs1.write(' <e1>'+event2[0][1:]+'</e1>')
                elif(event2[0][0]==' ' and event2[0][-1]==' '):
                    flag2=1
                    fs1.write(' <e1>'+event2[0][1:-1]+'</e1>')
                else:
                    flag2=1
                    fs1.write('<e1>'+event2[0][:-1]+'</e1>')
            else:
                fs1.write(s[0]+'<e1>')
                fs1.write(event2[0]+'</e1>')
            if(flag2==1):
                fs1.write(' '+event2[1])
                flag2=0
            else:
                fs1.write(event2[1])
        i=i+1
    else:
        fs1.write(text[i])
        i=i+1

fs.close()
fs1.close()                
            
            
            
    
    
    