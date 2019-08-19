# -*- coding: utf-8 -*-
"""
word_embeding训练
Created on Sun Oct 22 09:15:22 2017
第一步
输入数据：pubmed语料库处理后的数据
目标：word_embeddiing并将词向量模型存为pkl
@author: Administrator
"""

import codecs
import pickle
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from gensim.models.keyedvectors import KeyedVectors
from TFNN.utils.io_util import read_lines

def get_sentence(sentence_tag):
    words = []
    for item in sentence_tag.split(' '):
        index = item.rindex('/')                                                                                                                                                                                                                                                                                
        words.append(item[:index])
    return ' '.join(words)
    
def extract_sentece():
    #将测试集和训练集的句子合在一起
    #lines = read_lines('./Data/corpus/training.seg.csv')
    #lines += read_lines('./Data/corpus/testing.seg.csv')
    lines=read_lines('F:\\PubMedSpider\\sample\\new_together.txt')
    #创建一个新的文本
    with codecs.open('F:\\PubMedSpider\\sample\\only_sentence.txt', 'w', encoding='utf-8') as file_w:
        for line in lines:
            #注意：这里line.index(' ')里面的符号根据得到词性标记的文本实际情况而定
            #index = line.index(' ')
            #word_tag = line[index+1:]
            word_tag=line
            file_w.write('%s\n' % get_sentence(word_tag))

            
def train():
    extract_sentece()
    #in_path的txt就是上面extract_sentence运行的结果
    #这里的word2vec的对象只针对纯文本（即未加词性标注的文本）
    in_path = 'F:\\PubMedSpider\\sample\\only_sentence.txt'
    out_path = 'F:\\code_project1\\word2vec.bin'
    # 训练模型
    
    #sg=1,用skip-gram构造的
    #window：表示当前词与预测词在一个句子中的最大距离是多少
    #workers参数控制训练的并行数
    #iter： 迭代次数，默认为5
    model = Word2Vec(
        sg=1, sentences=LineSentence(in_path),
        size=256, window=20, min_count=1, workers=4, iter=40)
    ## 以一种C语言可以解析的形式存储词向量
    model.wv.save_word2vec_format(out_path, binary=True)

    
def bin2pkl():
    model = KeyedVectors.load_word2vec_format('F:\\G\My_project\\Project1\\code\\300features_40minwords_10context.bin', binary=True)
    word_dict = {}
    for word in model.vocab:
        word_dict[word] = model[word]
    with open('F:\\code_project1\\word2vec.pkl', 'wb') as file_w:
        #pickle模块的序列化操作我们能够将程序中运行的对象信息保存到文件中去
        #将word_dict保存到file_w中/Data/embedding/word2vec.pkl
        pickle.dump(word_dict, file_w)
        print(file_w.name)

if __name__ == '__main__':
    train()
    #将词向量，模型以pkl的形式存储
    bin2pkl()


