# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 10:36:53 2017
第一步
输入数据：有词性标注的文本
目标：创建词性字典和词字典、词id-词向量矩阵、词性id-词性向量矩阵
@author: Administrator
"""

import os
import pickle
import numpy as np
import configurations as config
from TFNN.utils.data_util import create_dictionary
from TFNN.utils.io_util import read_lines
from time import time

def init_voc():
    """
    初始化voc
    """
    #TRAIN_PATH = './Data/corpus/training.seg.csv'
    #TRAIN_PATH = 'F:\\PubMedSpyder\\new_together.txt'
    lines = read_lines(config.TRAIN_PATH)
    #lines += read_lines(config.TEST_PATH)
    words = []  # 句子
    pos_tags = []  # 词性标记类型
    for line in lines:
        #index = line.index(',')
        #sentence = line[index+1:]
        sentence=line
        # words and tags
        words_tags = sentence.split(' ')
        words_temp, tag_temp = [], []
        for item in words_tags:
            r_index = item.rindex('/')#/是词与词性的界限
            word, tag = item[:r_index], item[r_index+1:]
            #分别构造词典和词性词典
            words_temp.append(word)
            tag_temp.append(tag)
        pos_tags.extend(tag_temp)
        words.extend(words_temp)
    # word voc
    #WORD_VOC_PATH是含词（注意不是词向量）的pkl文件
    #得到的字典是下标与单词的词典
    create_dictionary(
        words, config.WORD_VOC_PATH, start=config.WORD_VOC_START,
        min_count=1, sort=True, lower=True, overwrite=True)
    # tag voc
    #TAG_VOC_PATH是含词性（注意不是词向量）的pkl文件
    #TAG_VOC_START=1代表起始下标
    create_dictionary(
        pos_tags, config.TAG_VOC_PATH, start=config.TAG_VOC_START,
        sort=True, lower=False, overwrite=True)
    # label voc
    #在BIONLP中事件类型有九种（这里可以先理解为触发词类型，因为是由触发词直接得到的类型
    label_types = [str(i) for i in range(1, 10)]
    create_dictionary(
        label_types, config.LABEL_VOC_PATH, start=0, overwrite=True)
    
    
def init_word_embedding(path=None, overwrite=False):
    """
    初始化word embedding
    Args:
        path: 结果存放路径
    """
    if os.path.exists(path) and not overwrite:
        return
    #W2V_PATH是存储词向量的文件（由word2vec_train.py得到）
    with open(config.W2V_PATH, 'rb') as file:
        w2v_dict_full = pickle.load(file)#向量词典
    #WORD_VOC_PATH是词字典，由上面的init_embedding函数的得到（具体在creat_dictionary中)
    with open(config.WORD_VOC_PATH, 'rb') as file:
        w2id_dict = pickle.load(file)#id词典
        
#'''!!!!!!注意下面的WORD_VOC_START的值，根据实际情况变换!!!!!'''
    word_voc_size = len(w2id_dict.keys()) + config.WORD_VOC_START
    #构造0矩阵
    #W2V_DIM是词向量中每个词的维度，为256
    word_weights = np.zeros((word_voc_size, config.W2V_DIM), dtype='float32')
    #得到标号（词id）-词向量矩阵
    for word in w2id_dict:
        index = w2id_dict[word]  # 词的标号
        if word in w2v_dict_full:
            #用训练好的词向量构造word_weights矩阵
            word_weights[index, :] = w2v_dict_full[word]
        else:
            #若当前词不在w2v_dict_full中，则随机化产生
            random_vec = np.random.uniform(
                -0.25, 0.25, size=(config.W2V_DIM,)).astype('float32')
            word_weights[index, :] = random_vec
    # 写入pkl文件
    #path为W2V_TRAIN_PATH
    with open(path, 'wb') as file:
        #将包好词向量的矩阵word_weights 以二进制文件形式存储
        pickle.dump(word_weights, file, protocol=2)

        
def init_tag_embedding(path, overwrite=False):
    """
    初始化pos tag embedding（词性矩阵）
    Args:
        path: 结果存放路径
    """
    if os.path.exists(path) and not overwrite:
        return
    with open(config.TAG_VOC_PATH, 'rb') as file:
        tag_voc = pickle.load(file)
    tag_voc_size = len(tag_voc.keys()) + config.TAG_VOC_START
    tag_weights = np.random.normal(
        size=(tag_voc_size, config.TAG_DIM)).astype('float32')
    for i in range(config.TAG_VOC_START):
        tag_weights[i, :] = 0.
    with open(path, 'wb') as file:
        pickle.dump(tag_weights, file, protocol=2)
        
        
        
def init_embedding():
    """
    初始化embedding
    """
    if not os.path.exists(config.EMBEDDING_ROOT):
        os.mkdir(config.EMBEDDING_ROOT)
    #调用之前的两个函数
    # 初始化word embedding
    init_word_embedding(config.W2V_TRAIN_PATH, overwrite=True)
    # 初始化tag embedding
    init_tag_embedding(config.T2V_PATH, overwrite=True)
    
def demo():
    #得到词向量矩阵的维度
    with open(config.W2V_TRAIN_PATH, 'rb') as file:
        temp = pickle.load(file)
    print(temp.shape)
    
if __name__ == '__main__':
    t0 = time()

    init_voc()  # 初始化voc

    init_embedding()  # 初始化embedding

    demo()

    print('Done in %.1fs!' % (time()-t0))
