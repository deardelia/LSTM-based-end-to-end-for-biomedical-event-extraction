# coding: utf-8

# In[1]:


## REQUIREMENTS ##0

import sys, os, _pickle as pickle
import tensorflow as tf
import numpy as np
import nltk
from sklearn.metrics import f1_score
from load_data import load_embedding, load_voc, load_train_data,load_test_data
from triggerType_to_trigger import get_trigger
from path_extractor import lca, path_lca,seq
import _pickle
from TFNN.layers.EmbeddingLayer import Embedding

##actual dim is 200
word_embd_dim = 100                        # Dimension of embedding layer for words
pos_embd_dim = 25                          # Dimension of embedding layer for POS Tags
dep_embd_dim = 25                          # Dimension of embedding layer for Dependency Types
word_vocab_size = 5443656                   # Vocab size for Words
pos_vocab_size = 10                        # Vocab size for POS Tags
dep_vocab_size = 21                        # Vocab size for Dependency Types
relation_classes = 7   #!!!modify                   # No. of Relation Classes
state_size = 100                           # Dimension of States of LSTM-RNNs
batch_size = 10       #batch_size modify                     # Batch Size for training
max_len_seq = 100                           # Maximum length of sentences
max_len_path = 20                          # Maximum length of lca paths
max_num_child = 20                         # Maximum no. of childrens in Dependency Tree
lambda_l2 = 0.0001                         # lambda of l2-regulaizer
init_learning_rate = 0.001                 # Initial Learning Rate
decay_steps = 2000                         # Decay Steps for Learning Rate
decay_rate = 0.96                          # Decay Rate for Learning Rate
gradient_clipping = 10                     # Size of Gradient Clipping 
hidden_size=100
class_type=1
training_count=16000
## INPUT ##
with tf.name_scope("input"):
    # Length of the whole sequence
    fp_length = tf.placeholder(tf.int32, shape=[batch_size], name="fp_length")
    # Words and POS Tags in sequence
    fp = tf.placeholder(tf.int32, [batch_size, 2, max_len_seq], name="full_path")
    # Length of both LCA Paths
    sp_length = tf.placeholder(tf.int32, shape=[batch_size, 2], name="sp_length")
    # Dependency Types in LCA Paths
    sp = tf.placeholder(tf.int32, [batch_size, 2, max_len_path], name="shortest_path")
    # Position of words in LCA Paths in whole sequence
    sp_pos = tf.placeholder(tf.int32, [batch_size, 2, max_len_path], name="sp_pos")
    # Position in whole sequence of the children in Dependency Tree of words in LCA Paths 
    sp_childs = tf.placeholder(tf.int32, [batch_size, 2, max_len_path, max_num_child], name="sp_childs")
    # No. of children in Dependency Tree of words in LCA Paths
    sp_num_childs = tf.placeholder(tf.int32, [batch_size, 2, max_len_path], name="sp_num_childs")
    # True Relation btw the entities
    relation = tf.placeholder(tf.int32, [batch_size], name="relation")
    # Hot vector of true entities
    y_entity = tf.placeholder(tf.int32, [batch_size, 1], name="y_enity")
	

## EMBEDDING LAYER ##
# Embedding Layer of Words 
with tf.name_scope("word_embedding"):
    W = tf.Variable(tf.constant(0.0, shape=[word_vocab_size, word_embd_dim]), name="W")
    embedding_placeholder = tf.placeholder(tf.float32,[word_vocab_size, word_embd_dim])
    embedding_init = W.assign(embedding_placeholder)#
    embd_fp_word = tf.nn.embedding_lookup(W,fp[:,0])
    word_embedding_saver = tf.train.Saver({"word_embedding/W": W})
# Embedding Layer of POS Tags #####create randomly
with tf.name_scope("pos_embedding"):
    W = tf.Variable(tf.random_uniform([pos_vocab_size, pos_embd_dim]), name="W")
    embd_fp_pos = tf.nn.embedding_lookup(W, fp[:,1])
    pos_embedding_saver = tf.train.Saver({"pos_embedding/W": W})
# Embedding layer as input for Forward sequential LSTM-RNNs
embd_fp = tf.concat([embd_fp_word, embd_fp_pos], axis=2)
# Embedding layer as input for Backward sequential LSTM-RNNs
embd_fp_rev = tf.reverse(embd_fp, [1])


# Conditions for while loops
def cond1(i, const, steps, *agrs):
    return i< steps
def cond2(i, steps, *agrs):
    return i< steps
def cond3(i,*agrs):
    return i<100
# Initial Hidden and Cell States for Sequential LSTMs
init_state_seq = tf.zeros([2, 1, state_size])
x = tf.constant(0)


# In[2]:
## SEQUENCE LAYER ##
# Function for initializing Sequential LSTM-RNN
def lstm_seq_init(channel, embedding_dim, state_size):
    init_const = tf.zeros([1, state_size])
    with tf.variable_scope(channel):        
        # Input Gate's weigths and bias
        W_i = tf.get_variable("W_i",shape=[embedding_dim, state_size] ,initializer=tf.contrib.layers.xavier_initializer())
        U_i = tf.get_variable("U_i",shape=[state_size, state_size] ,initializer=tf.contrib.layers.xavier_initializer())
        b_i = tf.get_variable("b_i", initializer=init_const)
        # Forget Gate's weigths and bias
        W_f = tf.get_variable("W_f",shape=[embedding_dim, state_size] ,initializer=tf.contrib.layers.xavier_initializer())
        U_f = tf.get_variable("U_f",shape=[state_size, state_size] ,initializer=tf.contrib.layers.xavier_initializer())
        b_f = tf.get_variable("b_f", initializer=init_const)
        # Output Gate's weigths and bias
        W_o = tf.get_variable("W_o",shape=[embedding_dim, state_size] ,initializer=tf.contrib.layers.xavier_initializer())
        U_o = tf.get_variable("U_o",shape=[state_size, state_size] ,initializer=tf.contrib.layers.xavier_initializer())
        b_o = tf.get_variable("b_o", initializer=init_const)
        W_g = tf.get_variable("W_g",shape=[embedding_dim, state_size] ,initializer=tf.contrib.layers.xavier_initializer())
        U_g = tf.get_variable("U_g",shape=[state_size, state_size] ,initializer=tf.contrib.layers.xavier_initializer())
        b_g = tf.get_variable("b_g", initializer=init_const)
# Intialized Forward Sequential LSTM-RNN
lstm_seq_init("lstm_fw", word_embd_dim + pos_embd_dim, state_size)
# Intialized Backward Sequential LSTM-RNN
lstm_seq_init("lstm_bw", word_embd_dim + pos_embd_dim, state_size)
# Function for running Sequence LSTM 
def lstm_seq(input_embd, seq_len, scope):    
    # While Loop body for running over the sequence
    def body(j, const, steps, input_embd, states_seq, states_series):
        inputs = tf.expand_dims(input_embd[j], [0])        
        # Hidden State of LSTM-RNN
        hs = states_seq[0]        
        # Cell State of LSTM-RNN
        cs = states_seq[1]        
        # Hidden State Series
        hs_ = states_series[0]       
        # Cell State Series
        cs_ = states_series[1]       
        with tf.variable_scope(scope, reuse=True):
            # Calling the Variables
            W_i = tf.get_variable("W_i")
            U_i = tf.get_variable("U_i")
            b_i = tf.get_variable("b_i")
            W_f = tf.get_variable("W_f")
            U_f = tf.get_variable("U_f")
            b_f = tf.get_variable("b_f")
            W_o = tf.get_variable("W_o")
            U_o = tf.get_variable("U_o")
            b_o = tf.get_variable("b_o")
            W_g = tf.get_variable("W_g")
            U_g = tf.get_variable("U_g")
            b_g = tf.get_variable("b_g")            
            input_gate = tf.sigmoid(tf.matmul(inputs, W_i) + tf.matmul(hs, U_i) + b_i)
            forget_gate = tf.sigmoid(tf.matmul(inputs, W_f) + tf.matmul(hs, U_f) + b_f)
            output_gate = tf.sigmoid(tf.matmul(inputs, W_o) + tf.matmul(hs, U_o) + b_o)
            gt = tf.tanh(tf.matmul(inputs, W_g) + tf.matmul(hs, U_g) + b_g)
            cs = input_gate * gt + forget_gate * cs
            hs = output_gate * tf.tanh(cs)            
            # Concating Hidden State Series and Hidden State
            hs_ = tf.cond(tf.equal(j, const), lambda: hs, lambda: tf.concat([hs_, hs], 0))            
            # Concating Cell State Series and Cell State
            cs_ = tf.cond(tf.equal(j, const), lambda: cs, lambda: tf.concat([cs_, cs], 0))   
            # Stacking Hidden and Cell State
            states_seq = tf.stack([hs, cs], axis=0)            
            # Stacking Hidden and Cell State Series
            states_series = tf.stack([hs_, cs_], axis=0)            
            return j+1, const, steps, input_embd, states_seq, states_series    
    # Running While Loop over the Sequence
    _, _, _, _, _, state_series_seq = tf.while_loop(cond1, body, 
            [0, 0, seq_len, input_embd, init_state_seq, init_state_seq],
            shape_invariants=[x.get_shape(), x.get_shape(), x.get_shape(), input_embd.get_shape(),
            init_state_seq.get_shape(), 
            tf.TensorShape([2, None, state_size])])    
    # Return State Series of Sequence LSTMs
    return state_series_seq
# Computing the Sequence Layer
states_series_fw = []
states_series_bw = []
hidden_states_seq = []
for b in range(batch_size):
    seq_len = fp_length[b]
    input_embd = embd_fp[b]  
    # Running Forward Sequence LSTM
    states_series_fw.append(lstm_seq(input_embd, seq_len, "lstm_fw"))    
    input_embd = embd_fp_rev[b]  
    # Running Backward Sequence LSTM
    states_series_bw.append(tf.reverse(lstm_seq(input_embd, seq_len, "lstm_bw"), [1]))    
    # Concating Hidden States of both Forward and Backward Seq LSTMs
    hidden_states_seq.append(tf.concat([states_series_fw[b][0], states_series_bw[b][0]], 1))


## ENTITY DETECTION ## 
# Hidden Layer After Sequence LSTM
"""with tf.name_scope("hidden_layer_seq"):
    W = tf.Variable(tf.truncated_normal([200, 100], -0.1, 0.1), name="W")
    b = tf.Variable(tf.zeros([100]), name="b")   
    y_hidden_layer = []
    y_hl = tf.zeros([1, 100])
    lstmCell = tf.contrib.rnn.BasicLSTMCell(hidden_size)
    lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=1)	
    for batch in range(batch_size):
        s_seq = tf.expand_dims(hidden_states_seq[batch], 1)       
        # Looping over the equence for computing Hidden Layer
        def matmul_hl(j, const, steps, input_seq, out_seq):
            temp = tf.tanh(tf.matmul(input_seq[j], W) + b)
            out_seq = tf.cond(tf.equal(j, const), lambda: temp, lambda: tf.concat([out_seq, temp], 0))
            return j+1, const, steps, input_seq, out_seq        
        _, _, _, _, output_seq = tf.while_loop(cond1, matmul_hl, 
                                [0, 0, fp_length[batch], s_seq, y_hl],
                                shape_invariants=[x.get_shape(), x.get_shape(), x.get_shape(), 
                                s_seq.get_shape(), tf.TensorShape([None, 100])])        
        y_hidden_layer.append(output_seq)"""
def bi_lstm(X_inputs,X_tag_inputs):
    nil_vars = set()
    word_embed_layer = Embedding(
                                 params=word_weights, ids=X_inputs,
                                 keep_prob=1.0, name='word_embed_layer')
    tag_embed_layer = Embedding(
            params=tag_weights, ids=X_tag_inputs,
            keep_prob=1.0, name='tag_embed_layer')
    nil_vars.add(word_embed_layer.params.name)
    nil_vars.add(tag_embed_layer.params.name)
    sentence_input = tf.concat(
		values=[word_embed_layer.output, tag_embed_layer.output], axis=2)
    inputs=sentence_input
    lstmCell = tf.contrib.rnn.BasicLSTMCell(hidden_size)
    lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=1)
    result, _ = tf.nn.dynamic_rnn(lstmCell, inputs, dtype=tf.float32)
    result = tf.transpose(result, [1, 0, 2])
    output = tf.gather(result, int(result.get_shape()[0]) - 1)  
    return output
	
with tf.variable_scope('bilstm_Inputs'):
    X_inputs = tf.placeholder(tf.int32, shape=(None, max_len_seq), name='X_input')
    X_tag_inputs = tf.placeholder(tf.int32, shape=(None, max_len_seq), name='X_input')  
    labels_train = tf.placeholder(tf.int32, shape=(None,), name='labels_train')   
bilstm_output = bi_lstm(X_inputs,X_tag_inputs)
with tf.variable_scope('bilstm_outputs'):
    class_num=2
    weights = tf.Variable(tf.truncated_normal([hidden_size, class_num], stddev=0.1))
    bias = tf.Variable(tf.constant(0.1, shape=[class_num]))
    y_pred = tf.nn.sigmoid(tf.add(tf.matmul(bilstm_output, weights) , bias))
correct_prediction = tf.equal(tf.cast(tf.argmax(y_pred, 1), tf.int32), labels_train)
correct_prediction_=tf.cast(correct_prediction, tf.float32)
y_type=correct_prediction_
Y_softmax=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = (labels_train), logits = y_pred))
 
with tf.name_scope("loss_seq"):
    loss_seq = tf.reduce_mean(Y_softmax) 
#get trigger
prediction_test=[]
for i in y_type:
    prediction_test.append(int(float(i)))
#triggers中存储当前batch_size个句子的触发词(针对某一类型而言),places存储该触发词在这个句子中所在的位置
triggrs,places=get_trigger(class_type,prediction_test,batch_size)
#use triggers to get word_path2, dep_path2,  pos_tags_path2, pos_path2,  childs_path2
for i in range(length):   
    try:
        parse_tree = dep_parser.raw_parse(sentences[i])
        for trees in parse_tree:
            tree = trees
		#!!!!!!node1
            node1 = tree.nodes[e1[i]+1]
            node2 = tree.nodes[places[i]+1]
        if node1['address']!=None and node2['address']!=None:
            #print(i, "success")
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
    except AssertionError:
        print(i, "error")
		
		
	
## DEPENDENCY LAYER ##
# Embedding Layer of Dependency Types #####create randomly
with tf.name_scope("dep_embedding"):
    W = tf.Variable(tf.random_uniform([dep_vocab_size, dep_embd_dim]), name="W")
    embd_sp = tf.nn.embedding_lookup(W, sp)
    dep_embedding_saver = tf.train.Saver({"dep_embedding/W": W})
# Function for initializing Tree Structured LSTM-RNN
def lstm_dep_init(channel, dep_input_size, state_size):
    init_const = tf.zeros([1, state_size])
    with tf.variable_scope(channel):
        
        # Input Gate's weigths and bias
        W_i = tf.get_variable("W_i", shape=[dep_input_size, state_size], initializer=tf.contrib.layers.xavier_initializer())
        U_i = tf.get_variable("U_i", shape=[state_size, state_size], initializer=tf.contrib.layers.xavier_initializer())
        b_i = tf.get_variable("b_i", initializer=init_const)
        
        # Input Gate's weights for Children in Dependency Tree
        U_it = tf.get_variable("U_it", shape=[state_size, state_size], initializer=tf.contrib.layers.xavier_initializer())

        
        # Forget Gate's weigths and bias
        W_f = tf.get_variable("W_f", shape=[dep_input_size, state_size], initializer=tf.contrib.layers.xavier_initializer())
        U_f = tf.get_variable("U_f", shape=[state_size, state_size], initializer=tf.contrib.layers.xavier_initializer())
        b_f = tf.get_variable("b_f", initializer=init_const)
        
        # Forget Gate for Children in LCA Path's weigths
        U_fsp = tf.get_variable("U_fsp", shape=[2, state_size, state_size], initializer=tf.contrib.layers.xavier_initializer())
        
        # Forget Gate for Children in Dependency Tree's weigths
        U_ffp = tf.get_variable("U_ffp", shape=[max_num_child, state_size, state_size], initializer=tf.contrib.layers.xavier_initializer())

        
        # Output Gate's weigths and bias
        W_o = tf.get_variable("W_o", shape=[dep_input_size, state_size], initializer=tf.contrib.layers.xavier_initializer())
        U_o = tf.get_variable("U_o", shape=[state_size, state_size], initializer=tf.contrib.layers.xavier_initializer())
        b_o = tf.get_variable("b_o", initializer=init_const)
        
        # Output Gate's weights for Children in Dependency Tree
        U_ot = tf.get_variable("U_ot", shape=[state_size, state_size], initializer=tf.contrib.layers.xavier_initializer())
        W_u = tf.get_variable("W_u", shape=[dep_input_size, state_size], initializer=tf.contrib.layers.xavier_initializer())
        U_u = tf.get_variable("U_u", shape=[state_size, state_size], initializer=tf.contrib.layers.xavier_initializer())
        b_u = tf.get_variable("b_u", initializer=init_const)
        U_ut = tf.get_variable("U_ut", shape=[state_size, state_size], initializer=tf.contrib.layers.xavier_initializer())
# Dimension of input in Treestructured LSTM-RNN
dep_input_size = state_size * 2 + dep_embd_dim
# Initializing  Bottom-Up Treestructured LSTM
lstm_dep_init("lstm_btup", dep_input_size, state_size)
# Initializing Top-Down Treestructured LSTM
lstm_dep_init("lstm_tpdn", dep_input_size, state_size)
# Initial Hidden and Cell State for Treestructured LSTM
init_state = tf.zeros([2, 1, 1, state_size])
# Function for running TreeStructured LSTM
def lstm_dep(b, p, start, seq_len, input_embd, input_pos, input_childs, input_num_child, states_seq, init_state_dep, scope):  
    
    # While loop Body for running TreeStrucctured LSTM
    def loop_over_seq(index, const, steps, input_pos, input_embd, input_childs, input_num_child, states_seq, states_dep, states_series):
        
        # Input for the lstm dep
        inputs = tf.expand_dims(tf.concat([hidden_states_seq[b][input_pos[p][index]], input_embd[p][index]],0),0)
       
        # Children in Dependency Tree
        childs = input_childs[p][index]
        
        # No. of Children in Dependency Tree
        num_child = input_num_child[p][index]  
        
        # No. of Children in LCA Path
        num_child_sp = tf.shape(states_dep[0])[0]
        
        with tf.variable_scope(scope, reuse=True):
            # Calling the Variables
            W_i = tf.get_variable("W_i")
            U_i = tf.get_variable("U_i")
            b_i = tf.get_variable("b_i")
            U_it = tf.get_variable("U_it")

            W_f = tf.get_variable("W_f")
            U_f = tf.get_variable("U_f")
            b_f = tf.get_variable("b_f")
            U_fsp = tf.get_variable("U_fsp")
            U_ffp = tf.get_variable("U_ffp")
            
            W_o = tf.get_variable("W_o")
            U_o= tf.get_variable("U_o")
            b_o = tf.get_variable("b_o")
            U_ot = tf.get_variable("U_ot")

            W_u = tf.get_variable("W_u")
            U_u = tf.get_variable("U_u")
            b_u = tf.get_variable("b_u")
            U_ut = tf.get_variable("U_ut")    

            ## Computing Input, Forget, Output Gates 
            ## e.g. it = x*W + b + h*U
            it = tf.matmul(inputs, W_i) + b_i + tf.matmul(states_dep[0][0], U_i)
            ft = tf.matmul(inputs, W_f) + b_f + tf.matmul(states_dep[0][0], U_f)
            ot = tf.matmul(inputs, W_o) + b_o + tf.matmul(states_dep[0][0], U_o)
            ut = tf.matmul(inputs, W_u) + b_u + tf.matmul(states_dep[0][0], U_u)
            
            ## 
            def matmul(k, steps, it, ft, ot, ut):
                it += tf.matmul(states_dep[0][k], U_i)
                ft += tf.matmul(states_dep[0][k], U_f) 
                ot += tf.matmul(states_dep[0][k], U_o) 
                ut += tf.matmul(states_dep[0][k], U_u) 
                return k+1, steps, it, ft, ot, ut
            
            _, _, it, ft, ot, ut = tf.while_loop(cond2, matmul, [1, num_child_sp, it, ft, ot, ut])

            ## Looping over the children in Dependency Tree 
            ## No. of loops is equal to the number of children in the dependency tree
            def child_sum(k, steps, out, U): 
                ### Calculating out = out + h_child* U
                ### Suming for all the children
                out += tf.matmul(states_seq[0][childs[k]], U)
                return k+1, steps, out, U

            ## While loop with body as child_sum
            ## Computing input gate, output gate by suming the h*U for all the children in the dependency tree
            _, _, ht_i, _ = tf.while_loop(cond2, child_sum, [0, num_child, it, U_it])
            _, _, ht_o, _ = tf.while_loop(cond2, child_sum, [0, num_child, ot, U_ot])
            _, _, ht_u, _ = tf.while_loop(cond2, child_sum, [0, num_child, ut, U_ut])

            ## Sigmoid over the gates
            input_gate = tf.sigmoid(ht_i)
            output_gate = tf.sigmoid(ht_o)
            
            u_input = tf.tanh(ht_u)

            # Computing Cell State
            cell_state = input_gate * u_input 

            # Computing Forget Gates for Children in LCA Path and Adding it to Compute Cell State
            def cell_state_sp(k, steps, cell_state):
                _, _, f_sp, _ = tf.while_loop(cond2, child_sum, [0, num_child, ft, U_fsp[k]])
                cell_state += tf.sigmoid(f_sp) * states_dep[1][k]
                return k+1, steps, cell_state
            
            _, _, cell_state = tf.while_loop(cond2, cell_state_sp, [0, num_child_sp, cell_state])

            # Computing Forget Gates for Children in Dependency Tree and Adding it to Compute Cell State
            def cell_states_fp(k, steps, ctl):
                _, _, f_fp, _ = tf.while_loop(cond2, child_sum, [0, num_child, ft, U_ffp[k]])
                ctl += tf.sigmoid(f_fp) * states_seq[1][childs[k]]
                return k+1, steps, ctl

            # Cell State 
            _, _, cds = tf.while_loop(cond2, cell_states_fp, [0, num_child, cell_state])
            
            # Hidden State
            hds = tf.expand_dims(output_gate * tf.tanh(cds), 0)

            # Expanding one dimension 
            cds = tf.expand_dims(cds, 0)
            
            # Stacking Hidden and Cell State
            states_dep = tf.stack([hds, cds], axis=0)

            # Concating Hidden State Series
            hds_ = tf.cond(tf.equal(index, const), lambda: states_dep[0],
                           lambda: tf.concat([states_series[0], states_dep[0]], 0))
            # Concating Cell State Series
            cds_ = tf.cond(tf.equal(index, const), lambda: states_dep[1],
                           lambda: tf.concat([states_series[1], states_dep[1]], 0))

            # Stacking Hidden and Cell State Series
            states_series = tf.stack([hds_, cds_], axis=0)

        return index+1, const, steps, input_pos, input_embd, input_childs, input_num_child, states_seq, states_dep, states_series
    
    # Running While Loop over sequence(LCA Path)
    _, _, _, _, _, _, _, _, _, states_series_dep = tf.while_loop(cond1, loop_over_seq,[start, start, seq_len,
                            input_pos, input_embd, input_childs, input_num_child, states_seq, init_state_dep,
                            init_state], shape_invariants=[x.get_shape(), x.get_shape(), x.get_shape(),
                            input_pos.get_shape(), input_embd.get_shape(), input_childs.get_shape(),
                            input_num_child.get_shape(), states_seq.get_shape(),
                            tf.TensorShape([2, None, 1, state_size]), tf.TensorShape([2, None, 1, state_size])])
    
    # Return Hidden and Cell State Series of TreeStructured LSTMs
    return states_series_dep
lca_series_btup = []
dp_series_tpdn = []
for i in range(batch_size):    
    # Position of words in the sentence 
    input_pos = sp_pos[i]    
    # Dependency Embedding  
    input_embd = embd_sp[i]    
    # Children's position in the sentence
    input_childs = sp_childs[i]    
    # No. of children in Dependency tree
    input_num_child = sp_num_childs[i]   
    # Reverse Sequenece
    input_pos_rev = tf.reverse(sp_pos[i], [1])
    input_embd_rev = tf.reverse(embd_sp[i], [1])
    input_childs_rev = tf.reverse(sp_childs[i], [1])
    input_num_child_rev = tf.reverse(sp_num_childs[i], [1])   
    # Expanding Dimension of States of Sequence LSTMs
    states_seq_fw = tf.expand_dims(states_series_fw[i], 2)
    states_seq_bw = tf.expand_dims(states_series_bw[i], 2)   
    # Comuting States from 1st entity upto LCA (Bottom-Up)
    s1 = lstm_dep(i, 0, 0, sp_length[i][0]-1, input_embd, input_pos, input_childs, input_num_child,
                 states_seq_fw, init_state, "lstm_btup")   
    # Last State in Bottom-Up which will serve as previous state for LCA
    lca_btup = tf.cond(sp_length[i][0]>1, lambda: s1[:,sp_length[i][0]-2], lambda: init_state[:,0])
    # Computing States from 2nd entity upto LCA (Bottom-Up)
    s2 = lstm_dep(i, 1, 0, sp_length[i][1], input_embd_rev, input_pos_rev, input_childs_rev, input_num_child_rev,
                 states_seq_bw, init_state, "lstm_btup")    
    # Stacking Last State from both Bottom-Up Trees which will serve as previous state for LCA
    lca_btup = tf.cond(sp_length[i][1]>0, lambda: tf.stack([lca_btup, s2[:,sp_length[i][1]-1]],axis=1), lambda: tf.expand_dims(lca_btup, 1))
    # Computing State for LCA (Bottom Up)
    lca_series_btup.append(lstm_dep(i, 0, sp_length[i][0]-1, sp_length[i][0], input_embd, input_pos, input_childs, input_num_child, states_seq_fw, lca_btup, "lstm_btup")[0,0])    
    # Computing State for LCA (Top Down)
    lca_tpdn = lstm_dep(i, 0, 0, 1, input_embd_rev, input_pos_rev, input_childs_rev, input_num_child_rev, 
                        states_seq_bw, init_state, "lstm_tpdn")
    # Computing States from LCA to 1st entity (Top-Down)
    dp1 = lstm_dep(i, 0, 1, sp_length[i][0], input_embd_rev, input_pos_rev, input_childs_rev, input_num_child_rev,
                   states_seq_bw, lca_tpdn, "lstm_tpdn")[0,-1]    
    # dp1 has State of 1st entity (Top-Down)
    dp1 = tf.cond(sp_length[i][0]>1, lambda: dp1, lambda: lca_tpdn[0][0])
    # Computing States from LCA to 2nd entity (Top-Down)
    dp2 = lstm_dep(i, 1, 0, sp_length[i][1], input_embd, input_pos, input_childs, input_num_child,
                   states_seq_fw, lca_tpdn, "lstm_tpdn")[0,-1]    
    # dp2 has State of 2nd entity (Top-Down)
    dp2 = tf.cond(sp_length[i][1]>0, lambda: dp2, lambda: lca_tpdn[0][0])
    # Concating the States of 1st and 2nd Entities (Top-Down)
    dp_series_tpdn.append(tf.concat([dp1, dp2], 1))        
# Concating the LCA (Bottom-Up) State and Entities (Top-Down) States
for i in range(batch_size):    
    temp = tf.concat([lca_series_btup[i], dp_series_tpdn[i]], 1)
    if(i==0):
        dp_series = temp
    else:
        dp_series = tf.concat([dp_series,temp], axis=0)



   
	
## RELATION CLASSIFICATION ##
# Hidden Layer in Dependency Layer (after Tree Structured LSTMs)
with tf.name_scope("hidden_layer_dep"):
    W = tf.Variable(tf.truncated_normal([300, 100], -0.1, 0.1), name="W")
    b = tf.Variable(tf.zeros([100]), name="b")
    y_p = tf.tanh(tf.matmul(dp_series, W) + b)
# Dropout for Hidden Layer in Dependency Layer 
with tf.name_scope("dropout_hidden_dep"):
        y_p_drop = tf.nn.dropout(y_p, 0.3)   
# SoftMax Layer in Dependency Layer
with tf.name_scope("softmax_layer_dep"):
    W = tf.Variable(tf.truncated_normal([100, relation_classes], -0.1, 0.1), name="W")
    b = tf.Variable(tf.zeros([relation_classes]), name="b")
    logits = tf.matmul(y_p_drop, W) + b    
    predictions_dep = tf.argmax(logits, 1)
# Loss for Dependency Layer
with tf.name_scope("loss_dep"):
    loss_dep = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=relation))
		
# All Trainable Variables
tv_all = tf.trainable_variables()
tv_regu = []
# Variables that are not regularised
non_reg = ["word_embedding/W:0","pos_embedding/W:0",'dep_embedding/W:0']
for t in tv_all:
    if t.name not in non_reg:
        if t.name.find('b_')==-1:
            if t.name.find('b:')==-1:
                tv_regu.append(t)
# Total Loss
with tf.name_scope("total_loss"):
    l2_loss = lambda_l2 * tf.reduce_sum([ tf.nn.l2_loss(v) for v in tv_regu ])
    total_loss = l2_loss + loss_seq + loss_dep
# Global Steps for Entity Training and Relation Classification
global_step_seq = tf.Variable(0, trainable=False, name="global_step_seq")
global_step_dep = tf.Variable(0, trainable=False, name="global_step_dep")
# Learning Rates for Entity Training and Relation Classification
learning_rate_seq = tf.train.exponential_decay(init_learning_rate, global_step_seq, decay_steps, decay_rate, staircase=True)
learning_rate_dep = tf.train.exponential_decay(init_learning_rate, global_step_dep, decay_steps, decay_rate, staircase=True)
# Optimzier for Loss of Sequence Layer
optimizer_seq = tf.train.AdamOptimizer(learning_rate_seq).minimize(loss_seq, global_step=global_step_seq)
# Optimizer for Total loss
optimizer = tf.train.AdamOptimizer(learning_rate_dep)
# Gradients and Variables for Total Loss
grads_vars = optimizer.compute_gradients(total_loss)
for g, v in grads_vars:
    if(g==None):
        print(g, v)        
# Clipping of Gradients
clipped_grads = [(tf.clip_by_norm(grad, gradient_clipping), var) for grad, var in grads_vars]
# Training Optimizer for Total Loss
train_op = optimizer.apply_gradients(clipped_grads, global_step=global_step_dep)
# Summary 
grad_summaries = []
for g, v in grads_vars:
    if g is not None:
        grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
        sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
        grad_summaries.append(grad_hist_summary)
        grad_summaries.append(sparsity_summary)
grad_summaries_merged = tf.summary.merge(grad_summaries)
loss_seq_summary = tf.summary.scalar("loss_seq", loss_seq)
loss_dep_summary = tf.summary.scalar("loss_dep", loss_dep)
total_loss_summary = tf.summary.scalar("total_loss", total_loss)
summary = tf.summary.merge_all()



# In[3]:
## VOCAB Preprocessing Functions ##
# Vocab for Words (Glove)
f = open('F:\\code_project1\\code2.0\\argument_detect\\vocab_glove', 'rb')
vocab = pickle.load(f)
f.close()
# Word to Vector 
word2id = dict((w, i) for i,w in enumerate(vocab))
id2word = dict((i, w) for i,w in enumerate(vocab))
unknown_token = "UNKNOWN_TOKEN"
word2id['UNKNOWN_TOKEN']=5443657
# Vocab for POS Tags
pos_tags_vocab = []
for line in open('F:\\code_project1\\code2.0\\argument_detect\\pos_tags.txt'):
        pos_tags_vocab.append(line.strip())
# Vocab for Dependency Types
dep_vocab = []
for line in open('F:\\code_project1\\code2.0\\argument_detect\\dependency_types.txt'):
    dep_vocab.append(line.strip())
# Vocab for Relation Classes
relation_vocab = []
for line in open('F:\\code_project1\\code2.0\\argument_detect\\relation_typesv3.txt'):
    relation_vocab.append(line.strip())
# Relation to Vector
rel2id = dict((w, i) for i,w in enumerate(relation_vocab))
id2rel = dict((i, w) for i,w in enumerate(relation_vocab))
# POS Tags to Vector
pos_tag2id = dict((w, i) for i,w in enumerate(pos_tags_vocab))
id2pos_tag = dict((i, w) for i,w in enumerate(pos_tags_vocab))
# Dependency Types to Vector
dep2id = dict((w, i) for i,w in enumerate(dep_vocab))
id2dep = dict((i, w) for i,w in enumerate(dep_vocab))
pos_tag2id['OTH'] = 9
id2pos_tag[9] = 'OTH'
dep2id['OTH'] = 20
id2dep[20] = 'OTH'
# Grouping POS Tags
JJ_pos_tags = ['JJ', 'JJR', 'JJS']
NN_pos_tags = ['NN', 'NNS', 'NNP', 'NNPS']
RB_pos_tags = ['RB', 'RBR', 'RBS']
PRP_pos_tags = ['PRP', 'PRP$']
VB_pos_tags = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
_pos_tags = ['CC', 'CD', 'DT', 'IN']
def pos_tag(x):
    if x in JJ_pos_tags:
        return pos_tag2id['JJ']
    if x in NN_pos_tags:
        return pos_tag2id['NN']
    if x in RB_pos_tags:
        return pos_tag2id['RB']
    if x in PRP_pos_tags:
        return pos_tag2id['PRP']
    if x in VB_pos_tags:
        return pos_tag2id['VB']
    if x in _pos_tags:
        return pos_tag2id[x]
    else:
        return 9
		
		
## DATA PREPROCESSING ##
# Function for preparing input for training or testing
def prepare_input1(words_seq, deps_seq, pos_tags_seq):
    # Size of the dataset
    length = len(words_seq)   
    # Removing None
    for i in range(length):
            words_seq[i] = [w for w in words_seq[i] if w !=None ]
            pos_tags_seq[i] = [w for w in pos_tags_seq[i] if w !=None ]

   # Converting Words, POS Tags, Dependency Types, Relation Classes to Vectors
    for i in range(length):
        for j, word in enumerate(words_seq[i]):
            word = word.lower()
            words_seq[i][j] = word if word in word2id else unknown_token 

    words_seq_id = np.ones([length, max_len_seq],dtype=int)
    pos_tags_seq_id = np.ones([length, max_len_seq],dtype=int)
      
    seq_len = []

    for i in range(length):        
        temp = []
        seq_len.append(len(words_seq[i]))
        for j, w in enumerate(words_seq[i]):
            words_seq_id[i][j] = word2id[w]       
        for j, w in enumerate(pos_tags_seq[i]):
            pos_tags_seq_id[i][j] = pos_tag(w)
                   
    return seq_len, words_seq_id, pos_tags_seq_id
"""seq_len:每个句子包含的单词数
#words_seq_id:14600*100的矩阵。每个元素为对应单词的值(为独热码，即单词在词典中的位置)
pos_tags_seq_id：14600*100的矩阵。每个元素为对应单词的词性标注
deps_seq_id:14600*100的矩阵。每个元素为对应单词
"""
def prepare_input(words_seq, deps_seq, pos_tags_seq, word_path1, word_path2, dep_path1, dep_path2, pos_tags_path1, pos_tags_path2, pos_path1,pos_path2, childs_path1, childs_path2, relations):
    # Size of the dataset
    length = len(words_seq)   
    # Position of words of LCA Paths in whole sentence 
    pos_path1 = [[i-1 for i in w] for w in pos_path1]
    pos_path2 = [[i-1 for i in w] for w in pos_path2]
    # Removing None
    for i in range(length):
            words_seq[i] = [w for w in words_seq[i] if w !=None ]
            deps_seq[i] = [w for w in deps_seq[i] if w !=None ]
            pos_tags_seq[i] = [w for w in pos_tags_seq[i] if w !=None ]
    # Hot Vector of Entities
    entity = np.zeros([length, max_len_seq])
    for i in range(length):
        entity[i][pos_path1[i][0]] = 1
        if(pos_path2[i]==[]):
            entity[i][pos_path1[i][-1]] = 1
        else:
            entity[i][pos_path2[i][0]] = 1
    len_path1 = []
    len_path2 = []
    num_child_path1 = np.ones([length, max_len_path],dtype=int)
    num_child_path2 = np.ones([length, max_len_path],dtype=int)
    # Length of LCA paths
    for w in word_path1:
        len_path1.append(len(w))
    for w in word_path2:
        len_path2.append(len(w))
    # No. of Children of words of LCA paths in sentence
    for i, w in enumerate(childs_path1):
        if(w!=[]):
            for j,c in enumerate(w):
                num_child_path1[i][j] = len(c)
        else:
            num_child_path1[i][0] = 0
    for i, w in enumerate(childs_path2):
        if(w!=[]):
            for j,c in enumerate(w):
                num_child_path2[i][j] = len(c)
        else:
            num_child_path2[i][0] = 0
    # Position of Children in sentence 
    for i in range(length):
        if(childs_path2[i]!=[]):
            for j, c in enumerate(childs_path2[i]):
                if(c == []):
                    childs_path2[i][j]  = [-1]
        else:
            childs_path2[i] = [[-1]]           
        if(childs_path1[i]!=[]):
            for j, c in enumerate(childs_path1[i]):
                if(c == []):
                    childs_path1[i][j]  = [-1]
        else:
            childs_path1[i] = [[-1]]
    # Replacing words not in vcab with unknown_token
    for i in range(length):
        if(word_path2[i]==[]):
            word_path2[i].append(unknown_token)
        if(dep_path2[i]==[]):
            dep_path2[i].append('OTH')
        if(pos_tags_path2==[]):
            pos_tags_path2[i].append('OTH')
    # Converting Words, POS Tags, Dependency Types, Relation Classes to Vectors
    for i in range(length):
        for j, word in enumerate(words_seq[i]):
            word = word.lower()
            words_seq[i][j] = word if word in word2id else unknown_token 
        for l, d in enumerate(deps_seq[i]):
            deps_seq[i][l] = d if d in dep2id else 'OTH'
        for j, word in enumerate(word_path1[i]):
            word = word.lower()
            word_path1[i][j] = word if word in word2id else unknown_token 
        for l, d in enumerate(dep_path1[i]):
            dep_path1[i][l] = d if d in dep2id else 'OTH'
        for j, word in enumerate(word_path2[i]):
            word = word.lower()
            word_path2[i][j] = word if word in word2id else unknown_token 
        for l, d in enumerate(dep_path2[i]):
            dep_path2[i][l] = d if d in dep2id else 'OTH'    
    word_path1_id = np.ones([length, max_len_path],dtype=int)
    word_path2_id = np.ones([length, max_len_path],dtype=int)    
    dep_path1_id = np.ones([length, max_len_path],dtype=int)
    dep_path2_id = np.ones([length, max_len_path],dtype=int)   
    pos_tags_path1_id = np.ones([length, max_len_path],dtype=int)
    pos_tags_path2_id = np.ones([length, max_len_path],dtype=int)    
    pos_path1_ = np.ones([length, max_len_path],dtype=int)
    pos_path2_ = np.ones([length, max_len_path],dtype=int)    
    childs_path1_ = np.ones([length, max_len_path, max_num_child],dtype=int)
    childs_path2_ = np.ones([length, max_len_path, max_num_child],dtype=int)
    for i in range(length):       
        temp = []
        for j, w in enumerate(pos_path1[i]):
            pos_path1_[i][j] = w       
        for j, w in enumerate(pos_path2[i]):
            pos_path2_[i][j] = w       
        for j,child in enumerate(childs_path1[i]):
            for k,c in enumerate(child):
                childs_path1_[i][j][k] = c -1               
        for j,child in enumerate(childs_path2[i]):
            for k,c in enumerate(child):
                childs_path2_[i][j][k] = c -1         
        for j, w in enumerate(word_path1[i]):
            word_path1_id[i][j]   = word2id[w]        
        for j, w in enumerate(pos_tags_path1[i]):
            pos_tags_path1_id[i][j] = pos_tag(w)       
        for j, w in enumerate(dep_path1[i]):
            dep_path1_id[i][j] = dep2id[w]       
        for j, w in enumerate(word_path2[i]):
            word_path2_id[i][j] = word2id[w]       
        for j, w in enumerate(pos_tags_path2[i]):
            pos_tags_path2_id[i][j] = pos_tag(w)      
        for j, w in enumerate(dep_path2[i]):
            dep_path2_id[i][j]  = dep2id[w]
    rel_ids = np.array([rel2id[rel] for rel in relations])
    
    return  len_path1, len_path2, pos_path1_, pos_path2_, dep_path1_id, dep_path2_id, childs_path1_, childs_path2_, num_child_path1, num_child_path2, rel_ids, entity 

# In[19]:


## GRAPH ##
sess = tf.Session()
# For initializing all the variables
sess.run(tf.global_variables_initializer())
# For Saving the model
saver = tf.train.Saver()
# For writing Summaries
summary_writer = tf.summary.FileWriter(summary_dir, sess.graph)




## TRAIN DATA PREPARATION ##
f = open('F:\\code_project1\\code2.0\\argument_detect\\train_pathsv3', 'rb')
words_seq, deps_seq, pos_tags_seq, _, _, _, _, _, _, _, _, _, _ = pickle.load(f)
seq_len, words_seq_id, pos_tags_seq_id=prepare_input1(words_seq, deps_seq, pos_tags_seq)
f.close()
relations = []
f = open('F:\\code_project1\\code2.0\\argument_detect\\train_data', 'rb')
#_, e1, _ = _pickle.load(f)
e1=[3,2,4,1,2,3,4,2,5,6]
f.close()
for line in open('F:\\code_project1\\code2.0\\argument_detect\\train_relationsv3.txt'):
    relations.append(line.strip().split()[1])
length = len(words_seq)
num_batches = int(length/batch_size)
word_weights, tag_weights = load_embedding()#矩阵形式
word_voc, tag_voc, label_voc = load_voc()#字典形式
sentences, tags, y_label = load_train_data(word_voc, tag_voc, label_voc,class_type,training_count)
#Xend_sentence, Xend_tag_test, yend_test = load_test_data(word_voc, tag_voc, label_voc,class_type,test_count)


## TRAINING ##
num_epochs = 10
prediction_test=[]
for i in range(num_epochs):    
    loss_per_epoch = 0   
    for j in range(num_batches):        
        s = j * batch_size
        end = (j+1) * batch_size        
        feed_dict={
			fp_length: seq_len[s:end],
			fp: [[words_seq_id[k], pos_tags_seq_id[k]] for k in range(s, end)],
			y_entity: y_label[s:end],
            labels_train:y_label[s:end],
			X_inputs:
		}
        word_path1_, word_path2_, dep_path1_, dep_path2_, pos_tags_path1_, pos_tags_path2_, pos_path1_, pos_path2_, childs_path1_, childs_path2_ =sess.run([word_path1, word_path2, dep_path1, dep_path2, pos_tags_path1, pos_tags_path2, pos_path1, pos_path2, childs_path1, childs_path2], feed_dict)
        len_path1, len_path2, pos_path1, pos_path2, dep_path1_id, dep_path2_id, childs_path1, childs_path2, num_child_path1, num_child_path2, rel_ids, entity = prepare_input(words_seq, deps_seq, pos_tags_seq, word_path1, word_path2, dep_path1, dep_path2, pos_tags_path1, pos_tags_path2, pos_path1,pos_path2, childs_path1, childs_path2, relations)
#loss？？？？？

        feed_dict = {            
            sp_length: [[len_path1[k], len_path2[k]] for k in range(s, end)],
            sp: [[dep_path1_id[k],dep_path2_id[k]] for k in range(s, end)],
            sp_pos : [[pos_path1[k], pos_path2[k]] for k in range(s, end)],
            sp_childs: [[childs_path1[k], childs_path2[k]] for k in range(s, end)],
            sp_num_childs: [[num_child_path1[k], num_child_path2[k]] for k in range(s, end)],
            relation: rel_ids[s:end],
            }
		
    # For entity pretraining 
#         _, _loss, step, _summary = sess.run([optimizer_seq, loss_seq, global_step_seq, summary], feed_dict)

    # For complete model training
        _, _loss, step, _summary ,y_result= sess.run([train_op, total_loss, global_step_dep, summary,y_type], feed_dict)
        
        # Suming the loss for each epoch
        loss_per_epoch +=_loss
        
        # Writing the summary
        summary_writer.add_summary(_summary, step)
        prediction_test.append(y_result)
        if(step%100==0):
            print("Steps:", step)
            
        if (j+1)%num_batches==0:
            print("Epoch:", i+1,"Step:", step, "loss:",loss_per_epoch/num_batches)
    
    # Saving the model      
    saver.save(sess, 'F:\\code_project1\\code2.0\\argument_detect\\model')
    print("Saved Model")
    fss=open("F:\\code_project1\\code2.0\\data_09\\test_data\\new_testdata\\prediction_test_newagain"+str(class_type)+".txt","r+")
    fss.write(str(prediction_test))
    fss.close()


# In[23]:
## TRAINING ACCURACY ##
all_predictions = []
for j in range(num_batches):
    s = j * batch_size
    end = (j+1) * batch_size

    feed_dict = {
        fp_length: seq_len[s:end],
        fp: [[words_seq_id[k], pos_tags_seq_id[k]] for k in range(s, end)],
        sp_length: [[len_path1[k], len_path2[k]] for k in range(s, end)],
        sp: [[dep_path1_id[k],dep_path2_id[k]] for k in range(s, end)],
        sp_pos : [[pos_path1[k], pos_path2[k]] for k in range(s, end)],
        sp_childs: [[childs_path1[k], childs_path2[k]] for k in range(s, end)],
        sp_num_childs: [[num_child_path1[k], num_child_path2[k]] for k in range(s, end)],
        relation: rel_ids[s:end],
        y_entity: y_label[s:end]}
        
    batch_predictions = sess.run(predictions_dep, feed_dict)
    all_predictions.append(batch_predictions)
y_pred = []
for i in range(num_batches):
    for pred in all_predictions[i]:
        y_pred.append(pred)
count = 0
for i in range(batch_size*num_batches):
    count += y_pred[i]==rel_ids[i]
accuracy = count/(batch_size*num_batches) * 100
f1 = f1_score(rel_ids[:batch_size*num_batches], y_pred, average='macro')*100
print("train accuracy", accuracy," F1 Score", f1)



#!!!!!!!!!!!!!!!create my own test data set
## TEST DATA PREPARATION ##
"""f = open('F:\\code_project1\\code2.0\\argument_detect\\test_pathsv3', 'rb')
words_seq, deps_seq, pos_tags_seq, word_path1, word_path2, dep_path1, dep_path2, pos_tags_path1, pos_tags_path2, pos_path1, pos_path2, childs_path1, childs_path2 = pickle.load(f)
f.close()

relations = []
for line in open('F:\\code_project1\\code2.0\\argument_detect\\test_relationsv3.txt'):
    relations.append(line.strip().split()[0])
    
length = len(words_seq)
num_batches = int(length/batch_size)

seq_len, words_seq_id, pos_tags_seq_id, deps_seq_id, len_path1, len_path2, pos_path1, pos_path2, dep_path1_id, dep_path2_id, childs_path1, childs_path2, num_child_path1, num_child_path2, rel_ids, entity = prepare_input(words_seq, deps_seq, pos_tags_seq, word_path1, word_path2, dep_path1, dep_path2, pos_tags_path1, pos_tags_path2, pos_path1,pos_path2, childs_path1, childs_path2, relations)

## TEST ACCURACY ##

all_predictions = []
for j in range(num_batches):
    s = j * batch_size
    end = (j+1) * batch_size

    feed_dict = {
        fp_length: seq_len[s:end],
        fp: [[words_seq_id[k], pos_tags_seq_id[k]] for k in range(s, end)],
        sp_length: [[len_path1[k], len_path2[k]] for k in range(s, end)],
        sp: [[dep_path1_id[k],dep_path2_id[k]] for k in range(s, end)],
        sp_pos : [[pos_path1[k], pos_path2[k]] for k in range(s, end)],
        sp_childs: [[childs_path1[k], childs_path2[k]] for k in range(s, end)],
        sp_num_childs: [[num_child_path1[k], num_child_path2[k]] for k in range(s, end)],
        relation: rel_ids[s:end],
        y_entity: entity[s:end]}
        
    batch_predictions = sess.run(predictions_dep, feed_dict)
    all_predictions.append(batch_predictions)
y_pred = []
for i in range(num_batches):
    for pred in all_predictions[i]:
        y_pred.append(pred)

count = 0
for i in range(batch_size*num_batches):
    count += y_pred[i]==rel_ids[i]
accuracy = count/(batch_size*num_batches) * 100

f1 = f1_score(rel_ids[:batch_size*num_batches], y_pred, average='macro')*100
print("test accuracy", accuracy," F1 Score", f1)"""







