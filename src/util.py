# encoding: utf-8
# author: xiongyuanpeng
# day: 2018-11-14
# day: 2019-1-2
from tensorlayer.layers.core import Layer
#import tensorflow.keras.backend as K
import tensorflow as tf
import tensorlayer as tl
from attention_tf import *
class MultiTasking:
    computing_type = {'L1':tf.abs, 'L2':tf.square}
    def __init__(self, group_names, regularizer = 'L2'):
        self.group_names = group_names
                
        self.loss_list = {}
        self.weight_list = {}   
        self.weight_sum = None
        self.mean_weight = {}
        self.total_sample = 0
        
        self.compute = self.computing_type[regularizer]
        
        self._init_group_weight()
        
    def _init_group_weight(self):
        
        for sample in self.group_names:
            self.add_weight(sample, tl.layers.get_variables_with_name(sample,True, True))
            
    def add_weight(self, sample, weights):
        if sample not in self.weight_list:
            self.total_sample += 1
            self.weight_list[sample] = weights        
            if self.weight_sum is None:
                self.weight_sum = weights
            else:
                for i in range(len(weights)):
                    self.weight_sum[i] = tf.add(self.weight_sum[i], weights[i])
        else:
            print('Type already exists!')
            self._update_weight(sample, weights)
            
    def _update_weight(self, sample, weights):
        print('Updating weight.')
        for i in range(len(weights)):
            self.weight_sum[i] = tf.add(self.weight_sum[i], weights[i]) - self.weight_list[sample][i]
        self.weight_list[sample] = weights
        
    def get_loss(self, sample):
        if self.total_sample is 0:
            print('[Warning] No weight in list!')
            return None
        # weight regularizer
        s = tf.constant(0.0)#tf.reduce_sum(tf.abs(hg19_list[0]))
        for i in range(len(self.weight_list[sample])):
            s = tf.add(s, tf.reduce_mean(self.compute(self.weight_list[sample][i])))
        
        # task similarity
        
        for i in range(len(self.weight_list[sample])):
            s = tf.add(s, tf.sqrt(tf.reduce_mean(tf.square(self.weight_list[sample][i] - tf.div(self.weight_sum[i], self.total_sample)))))

        return s
    
    def __pre_compute_loss(self):
        pass
        
class MultiWrapper:

    def __init__(self, group_setting):
        # building sample to group based on group_setting
        self.sample2group = {}
        self.groups = {}
        for key,item in group_setting.items():            
            self.groups[key] = MultiTasking(item)
            
            for i in item:
                self.sample2group[i] = key
                
    def get_loss(self, sample):
        
        return self.groups[self.sample2group[sample]].get_loss(sample)
        
        
class SelfAttentionLayer(Layer):

    def __init__(self, prev_layer, nb_head, size_per_head,act = None, name = 'attention'):
        
        super(SelfAttentionLayer, self).__init__(prev_layer = prev_layer,act =act, name=name)
        self.inputs = prev_layer.outputs
        print(self.inputs.shape)
        wh = int(self.inputs.shape[-1])
        with tf.variable_scope(name):
            
            #b = tf.get_variable(name = 'b_attention', shape = [32,])
            W1 = tf.get_variable(name='W1_attention', shape=[wh, nb_head * size_per_head])
            #b1 = tf.get_variable(name = 'b1_attention', shape = 1,])
            W2 = tf.get_variable(name='W2_attention', shape=[wh, nb_head * size_per_head])

            W = tf.get_variable(name='W_attention', shape=[wh, nb_head * size_per_head])

            Q = self._Dense(self.inputs, W, nb_head * size_per_head)

            Q = tf.reshape(Q, (-1, Q.shape[1], nb_head, size_per_head))
            print(Q.shape)
            Q = tf.transpose(Q, [0, 2, 1, 3])
            K = self._Dense(self.inputs, W1, nb_head * size_per_head)
            K = tf.reshape(K, (-1, K.shape[1], nb_head, size_per_head))
            K = tf.transpose(K, [0, 2, 1, 3])
            V = self._Dense(self.inputs, W2, nb_head * size_per_head)
            V = tf.reshape(V, (-1, V.shape[1], nb_head, size_per_head))
            V = tf.transpose(V, [0, 2, 1, 3])
            #print(V.shape)
            #计算内积，然后mask，然后softmax
            A = tf.matmul(Q, K, transpose_b=True) / tf.sqrt(float(size_per_head))
            A = tf.transpose(A, [0, 3, 2, 1])
            A = Mask(A, None, mode='add')
            A = tf.transpose(A, [0, 3, 2, 1])
            A = tf.nn.softmax(A)
            #输出并mask
            O = tf.matmul(A, V)
            #print(O.shape)
            O = tf.transpose(O, [0, 2, 1, 3])
            O = tf.reshape(O, (-1, O.shape[1], nb_head * size_per_head))
            O = Mask(O, None, 'mul')

            self.outputs = O#K.softmax(attmap)
        print(self.outputs.shape)
        self._add_layers(self.outputs)
        self._add_params([W, W1, W2])
        
    def _Dense(self, inputs, W, output_size):
        input_size = int(inputs.shape[-1])
        outputs = tf.matmul(tf.reshape(inputs, (-1, input_size)), W)# + b
        outputs = tf.reshape(outputs, \
                             tf.concat([tf.shape(inputs)[:-1], [output_size]], 0)
                            )
        #if seq_len != None:
        #    outputs = Mask(outputs, seq_len, 'mul')
        #print(outputs.shape)
        return outputs


dic = {'A':[1,0,0,0],'T':[0,1,0,0],'C':[0,0,1,0], 'G':[0,0,0,1],
      'R':[0.5,0,0,0.5], 'Y':[0,0.5,0.5,0,0],'M':[0.5,0,0.5,0],'K':[0,0.5,0,0.5],'S':[0,0,0.5,0.5],'W':[0.5,0.5,0,0],
       'H':[0.33,0.33,0.33,0],'B':[0,0.33,0.33,0.33],'V':[0.33,0,0.33,0.33],'D':[0.33,0.33,0,0.33],
      'N':[0.25, 0.25, 0.25, 0.25],}
'''
dic = {'A':1,'T':2,'C':3, 'G':4, 'N':5}
#      'R':5, 'Y':6,'M':7,'K':8,'S':9,'W':10,
#       'H':11,'B':12,'V':13,'D':14,
#d      'N':15,}
'''

def stringOnehot(line):
    x = [dic[ch] for ch in line]    
    return x
#sample_name = neg_directory  + 'TAIR10_neg_seqs_1000'

def readFileList(sample_name):
    '''
    read all data from files
    '''
    with open(sample_name,'r') as fp:            
        data = [[dic[ch] for ch in line.strip()] for line in fp.readlines()]    # 
    return data

def readList(lines):
    # change here
    
    data = [[dic[ch] for ch in line.strip()] for line in lines]
    #data = [[dic[ch] for ch in line.strip()[450:-450]] for line in lines]
    return data

def readFileNumpy(sample_name):
    data_np = None
    with open(sample_name,'r') as fp:   
        for line in fp.readlines():
            line = line.strip()
            y = np.ones((1,1001,4))
            for i in range(1001):
                y[0,i,:] = dic[line[i]]
            if data_np is None:
                data_np = y
            else:
                data_np = np.vstack([data_np, y])
    return data_np


from glob import glob

def cat_1000():
    files = glob('/home/xiongyuanpeng/Research/m6a_prediction/data/sequence_samples_downsample/*/*/*/*_1000')

    for f in files:
        with open(f,'r') as fp:
            fp1 = open(f+'_100','w')
            for line in fp.readlines():
                line = line.strip()[450:-450]
                fp1.write(line+'\n')


if __name__ == '__main__':
    cat_1000()

