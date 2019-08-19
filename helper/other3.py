#!/home/wenger/yuxinyi/code2.0
# -*- coding: utf-8 -*-
from load_data import load_embedding, load_voc, load_train_data,load_test_data
import time
from generator import BatchGenerator
from TFNN.layers.EmbeddingLayer import Embedding
from sklearn.model_selection import KFold
from triggerType_to_trigger import get_trigger


'''
For Chinese word segmentation.
'''


#############################1.load data   ######################################
class_type=3
training_count = 16796
test_count = 2570
word_weights, tag_weights = load_embedding()#矩阵形式
word_voc, tag_voc, label_voc = load_voc()#字典形式
sentences, tags, labels = load_train_data(word_voc, tag_voc, label_voc,class_type,training_count)
Xend_sentence, Xend_tag_test, yend_test = load_test_data(word_voc, tag_voc, label_voc,class_type,test_count)

#划分训练集，测试集（这里的y为词性tag

kf = KFold(n_splits=10)
train_indices, dev_indices = [], []
for train_index, dev_index in kf.split(labels):
    train_indices.append(train_index)
    dev_indices.append(dev_index)
for num in range(10):
    train_index, dev_index = train_indices[num], dev_indices[num]
    sentences_train, sentences_dev = sentences[train_index], sentences[dev_index]
    tags_train, tags_dev = tags[train_index], tags[dev_index]
    labels_train, labels_dev = labels[train_index], labels[dev_index]

"""kf = KFold(n_splits=10)
train_indices, test_indices = [], []
for train_index, test_index in kf.split(labels_train_):
    train_indices.append(train_index)
    test_indices.append(test_index)
for num in range(10):
    train_index, test_index = train_indices[num], test_indices[num]
    sentences_train, sentences_test = sentences_train_[train_index], sentences_train_[test_index]
    tags_train, tags_test = tags_train_[train_index], tags_train_[test_index]
    labels_train, labels_test = labels_train_[train_index], labels_train_[test_index]
"""
#得到最终的训练集测试集验证集（注意：这里没有tag_train）
X_train=sentences_train
X_tag_train=tags_train
y_train=labels_train
X_valid=sentences_dev
X_tag_valid=tags_dev
y_valid=labels_dev
X_test=Xend_sentence
X_tag_test=Xend_tag_test
y_test=yend_test

#得到batch数据
#data_test = BatchGenerator(Xend_sentence,Xend_tag_test, yend_test, shuffle=False)
print ('Creating the data generator ...')
data_train = BatchGenerator(X_train, X_tag_train,y_train, shuffle=True)
data_valid = BatchGenerator(X_valid, X_tag_valid,y_valid, shuffle=False)
data_test = BatchGenerator(X_test,X_tag_test, y_test, shuffle=False)
print ('Finished creating the data generator.')

###################################2.############################################
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
from tensorflow.contrib import rnn
import numpy as np

# ##################### config ######################
decay = 0.85
max_epoch = 5
max_max_epoch = 10
timestep_size = max_len =300          # 句子长度
vocab_size = 5159    # 样本中不同字的个数+1(padding 0)，根据处理数据的时候得到
input_size = embedding_size = 64       # 字向量长度
class_num = 2
hidden_size = 128    # 隐含层节点数
layer_num = 2        # bi-lstm 层数
max_grad_norm = 5.0  # 最大梯度（超过此值的梯度将被裁剪）

lr = tf.placeholder(tf.float32, [])
keep_prob = tf.placeholder(tf.float32, [])
batch_size = tf.placeholder(tf.int32, [])  # 注意类型必须为 tf.int32
#model_save_path = './bi-lstm.ckpt'  # 模型保存位置
model_save_path = 'F:\\code_project1\\code2.0\\'+'model'+str(class_type)+'\\bi-lstm.ckpt'  # 模型保存位置


with tf.variable_scope('embedding'):
    embedding = tf.get_variable("embedding", [vocab_size, embedding_size], dtype=tf.float32)

def weight_variable(shape):
    """Create a weight variable with appropriate initialization."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    """Create a bias variable with appropriate initialization."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def lstm_cell():
    cell = rnn.LSTMCell(hidden_size, reuse=tf.get_variable_scope().reuse)
    return rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)
         
def bi_lstm(X_inputs,X_tag_inputs):
    """build the bi-LSTMs network. Return the y_pred"""
    # X_inputs.shape = [batchsize, timestep_size]  ->  inputs.shape = [batchsize, timestep_size, embedding_size]
    #inputs = tf.nn.embedding_lookup(embedding, X_inputs)  
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
    

    """# ** 1.构建前向后向多层 LSTM
    cell_fw = rnn.MultiRNNCell([lstm_cell() for _ in range(layer_num)], state_is_tuple=True)
    cell_bw = rnn.MultiRNNCell([lstm_cell() for _ in range(layer_num)], state_is_tuple=True)
  
    # ** 2.初始状态
    initial_state_fw = cell_fw.zero_state(batch_size, tf.float32)
    initial_state_bw = cell_bw.zero_state(batch_size, tf.float32)  
    
    # 下面两部分是等价的
    # **************************************************************
    # ** 把 inputs 处理成 rnn.static_bidirectional_rnn 的要求形式
    # ** 文档说明
    # inputs: A length T list of inputs, each a tensor of shape
    # [batch_size, input_size], or a nested tuple of such elements.
    # *************************************************************
    # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    # inputs.shape = [batchsize, timestep_size, embedding_size]  ->  timestep_size tensor, each_tensor.shape = [batchsize, embedding_size]
    inputs = tf.unstack(inputs, timestep_size, 1)
    # ** 3.bi-lstm 计算（tf封装）  一般采用下面 static_bidirectional_rnn 函数调用。
     try:
         outputs, _, _ = rnn.static_bidirectional_rnn(cell_fw, cell_bw, inputs, 
                         initial_state_fw = initial_state_fw, initial_state_bw = initial_state_bw, dtype=tf.float32)
     except Exception: # Old TensorFlow version only returns outputs not states
         outputs = rnn.static_bidirectional_rnn(cell_fw, cell_bw, inputs, 
                         initial_state_fw = initial_state_fw, initial_state_bw = initial_state_bw, dtype=tf.float32)
     output = tf.reshape(tf.concat(outputs, 1), [-1, hidden_size * 2])
     output = tf.gather(result, int(result.get_shape()[0]) - 1)

     #***********************************************************"""
    
    # ***********************************************************
    # ** 3. bi-lstm 计算（展开）
    #inputs = tf.unstack(inputs, timestep_size, 1)
    """with tf.variable_scope('bidirectional_rnn'):
        # *** 下面，两个网络是分别计算 output 和 state 
        # Forward direction
        outputs_fw = list()
        state_fw = initial_state_fw
        with tf.variable_scope('fw'):
            for timestep in range(timestep_size):
                if timestep > 0:
                    tf.get_variable_scope().reuse_variables()
                (output_fw, state_fw) = cell_fw(inputs[:, timestep, :], state_fw)
                outputs_fw.append(output_fw)
        
        # backward direction
        outputs_bw = list()
        state_bw = initial_state_bw
        with tf.variable_scope('bw') as bw_scope:
            inputs = tf.reverse(inputs, [1])
            for timestep in range(timestep_size):
                if timestep > 0:
                    tf.get_variable_scope().reuse_variables()
                (output_bw, state_bw) = cell_bw(inputs[:, timestep, :], state_bw)
                outputs_bw.append(output_bw)
        # *** 然后把 output_bw 在 timestep 维度进行翻转
        # outputs_bw.shape = [timestep_size, batch_size, hidden_size]
        outputs_bw = tf.reverse(outputs_bw, [0])
        # 把两个oupputs 拼成 [timestep_size, batch_size, hidden_size*2]
        output = tf.concat([outputs_fw, outputs_bw], 2)
        output = tf.transpose(output, perm=[1,0,2])
        output = tf.reshape(output, [-1, hidden_size*2])"""
    # ***********************************************************
    return output # [-1, hidden_size*2]


with tf.variable_scope('Inputs'):
    X_inputs = tf.placeholder(tf.int32, shape=(None, max_len), name='X_input')
    X_tag_inputs = tf.placeholder(tf.int32, shape=(None, max_len), name='X_input')
    y_inputs = tf.placeholder(tf.int32, shape=(None,), name='y_input')   

bilstm_output = bi_lstm(X_inputs,X_tag_inputs)

with tf.variable_scope('outputs'):
    class_num=2
    weights = weight_variable([hidden_size, class_num]) 
    bias = bias_variable([class_num]) 
    y_pred = tf.nn.sigmoid(tf.add(tf.matmul(bilstm_output, weights) , bias))

# adding extra statistics to monitor
# y_inputs.shape = [batch_size, timestep_size]

correct_prediction = tf.equal(tf.cast(tf.argmax(y_pred, 1), tf.int32), tf.reshape(y_inputs, [-1]))
y_result=y_pred
#correct_prediction_ = tf.cast(correct_prediction,tf.int32)
correct_prediction_=tf.cast(correct_prediction, tf.float32)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = tf.reshape(y_inputs, [-1]), logits = y_pred))
#cost= tf.reduce_mean(-tf.reduce_sum(y_inputs * tf.log(y_pred) + (1 - y_inputs) * tf.log(1 - y_pred), axis=1))
#cost=-1*tf.reduce_mean(self.label_ph*tf.log(tf.clip_by_value(y_pred,1e-10,1.0)))
# ***** 优化求解 *******
tvars = tf.trainable_variables()  # 获取模型的所有参数
grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), max_grad_norm)  # 获取损失函数对于每个参数的梯度
optimizer = tf.train.AdamOptimizer(learning_rate=lr)   # 优化器

# 梯度下降计算
train_op = optimizer.apply_gradients( zip(grads, tvars),
    global_step=tf.contrib.framework.get_or_create_global_step())
print ('Finished creating the bi-lstm model.')


################################3.test############################################
def test_epoch(dataset):
    """Testing or valid."""
    _batch_size = 500
    fetches = [accuracy, cost]
    _y = dataset.y
    data_size = _y.shape[0]
    batch_num = int(data_size / _batch_size)
    start_time = time.time()
    _costs = 0.0
    _accs = 0.0
    for i in range(batch_num):
        X_batch,X_tag_batch, y_batch = dataset.next_batch(_batch_size)
        feed_dict = {X_inputs:X_batch,X_tag_inputs:X_tag_batch, y_inputs:y_batch, lr:1e-5, batch_size:_batch_size, keep_prob:1.0}
        _acc, _cost = sess.run(fetches, feed_dict)
        _accs += _acc
        _costs += _cost    
    mean_acc= _accs / batch_num     
    mean_cost = _costs / batch_num
    return mean_acc, mean_cost
    
    
def test_epoch_1(dataset):
    """Testing or valid."""
    _batch_size = 500
    fetches = [correct_prediction_,accuracy, cost,y_result]
    _y = dataset.y
    data_size = _y.shape[0]
    batch_num = int(data_size / _batch_size)
    start_time = time.time()
    _costs = 0.0
    _accs = 0.0
    prediction_result=[]
    #last_matrix=np.zeros([500,2])
    for i in range(batch_num):
        X_batch,X_tag_batch, y_batch = dataset.next_batch(_batch_size)
        feed_dict = {X_inputs:X_batch,X_tag_inputs:X_tag_batch, y_inputs:y_batch, lr:1e-5, batch_size:_batch_size, keep_prob:1.0}
        prediction_current, _acc, _cost,result_current = sess.run(fetches, feed_dict)
        n=0
        while n<len(prediction_current):
                prediction_result.append(prediction_current[n])
                n=n+1
        if(i==0):
            pred_result=result_current
        else:
            pred_result=np.concatenate((pred_result,result_current),axis=0)
        _accs += _acc
        _costs += _cost    
    mean_acc= _accs / batch_num     
    mean_cost = _costs / batch_num
    return prediction_result,mean_acc, mean_cost,pred_result





sess.run(tf.global_variables_initializer())
tr_batch_size = 128 
max_max_epoch = 6
display_num = 5  # 每个 epoch 显示是个结果
tr_batch_num = int(data_train.y.shape[0] / tr_batch_size)  # 每个 epoch 中包含的 batch 数
display_batch = int(tr_batch_num / display_num)  # 每训练 display_batch 之后输出一次
saver = tf.train.Saver(max_to_keep=10)  # 最多保存的模型数量
for epoch in range(max_max_epoch):
    _lr = 1e-4
    if epoch > max_epoch:
        _lr = _lr * ((decay) ** (epoch - max_epoch))
    print ('EPOCH %d， lr=%g' % (epoch+1, _lr))
    start_time = time.time()
    _costs = 0.0
    _accs = 0.0
    show_accs = 0.0
    show_costs = 0.0
    for batch in range(tr_batch_num): 
        fetches = [accuracy, cost, train_op]
        X_batch,X_tag_batch, y_batch = data_train.next_batch(tr_batch_size)
        feed_dict = {X_inputs:X_batch, X_tag_inputs:X_tag_batch,y_inputs:y_batch, lr:_lr, batch_size:tr_batch_size, keep_prob:0.5}
        _acc, _cost, _ = sess.run(fetches, feed_dict) # the cost is the mean cost of one batch
        _accs += _acc
        _costs += _cost
        show_accs += _acc
        show_costs += _cost
        if (batch + 1) % display_batch == 0:
            valid_acc, valid_cost = test_epoch(data_valid)  # valid
            print ('\ttraining acc=%g, cost=%g;  valid acc= %g, cost=%g ' % (show_accs / display_batch,
                                                show_costs / display_batch, valid_acc, valid_cost))
            show_accs = 0.0
            show_costs = 0.0
    mean_acc = _accs / tr_batch_num 
    mean_cost = _costs / tr_batch_num
    if (epoch + 1) % 3 == 0:  # 每 3 个 epoch 保存一次模型
        save_path = saver.save(sess, model_save_path, global_step=(epoch+1))
        print ('the save path is ', save_path)
    print ('\ttraining %d, acc=%g, cost=%g ' % (data_train.y.shape[0], mean_acc, mean_cost))
    print ('Epoch training %d, acc=%g, cost=%g, speed=%g s/epoch' % (data_train.y.shape[0], mean_acc, mean_cost, time.time()-start_time))      
# testing
print ('**TEST RESULT:')
prediction_test, test_acc, test_cost,pre_result = test_epoch_1(data_test)
print( '**Test %d, acc=%g, cost=%g' % (data_test.y.shape[0], test_acc, test_cost))
#print(array)
print('**Save the test result**')
#fs_result = open('./testresult.txt','w')
#fs_result.write('accuracy = %d  ,   test_cose = %d \n' %(test_acc_,test_cost_) )
#print('accuracy = %d  ,   test_cose = %d \n' %(test_acc_,test_cost_))
print('**form tigeer type get trigger****')
#trigger_words = get_trigger(class_type,prediction_test,test_count)

fss=open("F:\\code_project1\\code2.0\\data_09\\test_data\\new_testdata\\prediction_test_newagain"+str(class_type)+".txt","w")
fss.write(str(prediction_test))
fss.close()
