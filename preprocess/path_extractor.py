# coding: utf-8

import os
from nltk.parse import stanford
import nltk
from itertools import chain
import _pickle 
import os
from nltk.parse.stanford import StanfordDependencyParser



os.environ['STANFORD_PARSER'] = 'F:\\code_project1\\code2.0\\argument_detect\\jars\\stanford-parser.jar' 
os.environ['STANFORD_MODELS'] = 'F:\\code_project1\\code2.0\\argument_detect\\jars\\stanford-parser-3.5.0-models.jar' 
#为JAVAHOME添加环境变量  
java_path = "C:\\Program Files\\Java\\jdk-9.0.1\\bin\\java.exe"  
os.environ['JAVAHOME'] = java_path

# Dependency Tree
path_to_jar = "F:\\code_project1\\code2.0\\argument_detect\\stanford-parser-full-2017-06-09\\stanford-parser-3.8.0-models\\edu\\stanford\\nlp\\models\\lexparser\\englishPCFG.ser.gz"
dep_parser=StanfordDependencyParser(model_path=path_to_jar)



# In[3]:


def lca(tree, index1, index2):
    node = index1
    path1 = []
    path2 = []
    path1.append(index1)
    path2.append(index2)
    while(node != tree.root):
        node = tree.nodes[node['head']]
        path1.append(node)
    node = index2
    while(node != tree.root):
        node = tree.nodes[node['head']]
        path2.append(node)
    for l1, l2 in zip(path1[::-1],path2[::-1]):
        if(l1==l2):
            temp = l1
    return temp


# In[4]:


def path_lca(tree, node, lca_node):
    path = []
    path.append(node)
    while(node != lca_node):
        node = tree.nodes[node['head']]
        path.append(node)
    return path


# In[5]:


def seq(lca):
    l=[lca]
    for key in tree.nodes[lca]['deps']:
        for i in tree.nodes[lca]['deps'][key]:
            l.extend(seq(i))
    return l


# In[6]:
f = open('F:\\code_project1\\code2.0\\argument_detect\\train_data', 'rb')
sentences, e1, e2 = _pickle.load(f)
f.close()

length =10

words_seq = []
pos_tags_seq = []
deps_seq = []
word_path1 = []
word_path2 = []
dep_path1 = []
dep_path2 = []
pos_tags_path1 = []
pos_tags_path2 = []
childs_path1 = []
childs_path2 = []
pos_path1 = []
pos_path2 = []

for i in range(length) :
    
    word_path1.append(0)
    word_path2.append(0)
    dep_path1.append(0)
    dep_path2.append(0)
    pos_tags_path1.append(0)
    pos_tags_path2.append(0)
    words_seq.append(0)
    pos_tags_seq.append(0)
    deps_seq.append(0)
    childs_path1.append(0)
    childs_path2.append(0)
    pos_path1.append(0)
    pos_path2.append(0)

	
for i in range(length):
    
    try:
        parse_tree = dep_parser.raw_parse(sentences[i])
        for trees in parse_tree:
            tree = trees
	
        word2pos = dict((tree.nodes[k]['address'], j) for k,j in zip(tree.nodes, range(len(tree.nodes))))
        pos2word = dict((j, tree.nodes[k]['address']) for k,j in zip(tree.nodes, range(len(tree.nodes))))

        pos_tags_seq[i] = [tree.nodes[k]['tag'] for k in tree.nodes][1:]
        words_seq[i] = [tree.nodes[k]['word'] for k in tree.nodes][1:]
        deps_seq[i] = [tree.nodes[k]['rel'] for k in tree.nodes][1:]
        
        node1 = tree.nodes[e1[i]+1]
        node2 = tree.nodes[e2[i]+1]
        if node1['address']!=None and node2['address']!=None:
            print(i, "success")
            lca_node = lca(tree, node1, node2)
            path1 = path_lca(tree, node1, lca_node)
            path2 = path_lca(tree, node2, lca_node)[:-1]

            word_path1[i] = [p["word"] for p in path1]
            word_path2[i] = [p["word"] for p in path2]
            dep_path1[i] = [p["rel"] for p in path1]
            dep_path2[i] = [p["rel"] for p in path2]
            pos_tags_path1[i] = [p["tag"] for p in path1]
            pos_tags_path2[i] = [p["tag"] for p in path2]
            
            pos_path1[i] = [word2pos[node['address']] for node in path1]
            pos_path2[i] = [word2pos[node['address']] for node in path2]
            childs = [sorted(chain.from_iterable(node['deps'].values())) for node in path1]
            childs_path1[i] = [[word2pos[c] for c in child] for child in childs]
            childs = [sorted(chain.from_iterable(node['deps'].values())) for node in path2]
            childs_path2[i] = [[word2pos[c] for c in child] for child in childs]
        else:
            print(i, node1["address"], node2["address"])
    except AssertionError:
        print(i, "error")
		
		
file = open('F:\\code_project1\\code2.0\\argument_detect\\train_pathsv3', 'wb')
_pickle.dump([words_seq, deps_seq, pos_tags_seq, word_path1, word_path2, dep_path1, dep_path2, pos_tags_path1, pos_tags_path2, pos_path1, pos_path2, childs_path1, childs_path2], file)




