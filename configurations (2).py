# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 10:38:18 2017
第一步
（为词向量训练的程序提供所必要的宏定义）
@author: Administrator
"""

import os


# --- corpus ---
TRAIN_PATH = './Data/corpus/training.seg.csv'
TEST_PATH = './Data/corpus/testing.seg.csv'


# --- voc ---
"""VOC_ROOT = './Data/voc'
if not os.path.exists(VOC_ROOT):
    os.mkdir(VOC_ROOT)"""
WORD_VOC_PATH = 'F:\\code_project1\\code2.0\\voc\\word_voc.pkl'
WORD_VOC_START = 2
TAG_VOC_PATH = 'F:\\code_project1\\code2.0\\voc\\tag_voc.pkl'
TAG_VOC_START = 1
LABEL_VOC_PATH = "F:\\code_project1\\code2.0\\voc\\label_voc.pkl"


# --- embedding ---
W2V_DIM = 256
W2V_PATH = 'F:\\code_project1\\code2.0\\embedding\\word2vec.pkl'
#EMBEDDING_ROOT = './Data/embedding/'
#if not os.path.exists(EMBEDDING_ROOT):
 #   os.mkdir(EMBEDDING_ROOT)
W2V_TRAIN_PATH = "F:\\code_project1\\code2.0\\embedding\\word2v.pkl"
T2V_PATH = "F:\\code_project1\\code2.0\\embedding\\tag2v.pkl"
TAG_DIM = 64


# --- training param ---
MAX_LEN = 300
BATCH_SIZE = 64
NB_LABELS = 11
NB_EPOCH = 30
#！！！这里的KEEP_PROB，因为本身数据集不大
KEEP_PROB = 0.5
WORD_KEEP_PROB = 0.9
TAG_KEEP_PROB = 0.9
KFOLD = 10
