# encoding: utf-8
# author: xiongyuanpeng
# 2018-11-14
import tensorflow as tf
import numpy as np
from tensorlayer.layers import Conv2d, LambdaLayer, ConvLSTMLayer, BiRNNLayer, InputLayer, DenseLayer, FlattenLayer, PReluLayer
from tensorlayer.layers import * #TileLayer, ElementwiseLayer, ExpandDimsLayer, Conv1d,ConcatLayer, ElementwiseLayer,DropoutLayer,MaxPool1d
import tensorlayer as tl  
from matplotlib import pyplot as plt
from config import *
import logging
from util import *
logging.basicConfig(level=logging.INFO)
KERNEL_SIZE = 5
stddev = 1
def Selector(t_sequences, reuse = False):
    '''
    This parts plays an role as a selector of fixed position in genes, works like attention mechanism
    sequences: tf.place_holder([None, steps, embedding_dim])
    '''
    w_init = tf.random_normal_initializer(stddev=stddev)
    b_init = None
    g_init = tf.random_normal_initializer(1., stddev)
    act = lambda x: tf.nn.leaky_relu(x, 0.2)

    with tf.variable_scope("selector", reuse=reuse) as vs:
        # tl.layers.set_name_reuse(reuse) # remove for TL 1.8.0+
        sequences = InputLayer(t_sequences, name='in')
        # is it ok to add a embedding layer here
        # use strided convolution to decrease length of sequences
        return sequences,sequences 
        sequences = Conv1d(sequences, 32,KERNEL_SIZE , stride = 2, dilation_rate = 1, act = act, name = 'conv_500') # 500
        
        sequences = Conv1d(sequences, 32, KERNEL_SIZE, stride = 2, dilation_rate = 1, act = act, name = 'conv_250') # 250 
        
        sequences = Conv1d(sequences, 32, KERNEL_SIZE, stride = 2, dilation_rate = 1, act = act, name = 'conv_125') # 125
        
        #sequences = Conv1d(sequences, 32, KERNEL_SIZE, stride = 4, dilation_rate = 1, act = act, name = 'conv_63') # 125
        
        #sequences = Conv1d(sequences, 32, KERNEL_SIZE, stride = 4, dilation_rate = 1, act = act, name = 'conv_31') # 125
        
        
        # stacking 3 bi-directiona,l lstm here
        
        bi = BiRNNLayer(sequences, cell_fn = tf.contrib.rnn.LSTMCell, n_hidden = config.TRAIN.RNN_HIDDEN, n_steps = config.TRAIN.RNN_STEPS + 1, return_last = False, name = 'bi1')
        bi = PReluLayer(bi, channel_shared = True, name='prelu1')
        
        #bi = BiRNNLayer(bi, cell_fn = tf.contrib.rnn.LSTMCell, n_hidden = config.TRAIN.RNN_HIDDEN, n_steps = config.TRAIN.TIME_STEPS, return_last = False, name='bi2')
        #bi = PReluLayer(bi, channel_shared = True, name = 'prelu2')
        
        #bi = BiRNNLayer(bi, cell_fn = tf.contrib.rnn.LSTMCell, n_hidden = config.TRAIN.RNN_HIDDEN, n_steps = config.TRAIN.RNN_STEPS + 1, return_last = False, name = 'bi3')
        #bi = PReluLayer(bi, channel_shared = True, name='prelu3')
        # use last outputs of bi-lstm to generate attention
        
        features = FlattenLayer(bi, name='flatten_feature')
        
        # downsample was introduced for the overfitting issue
        sampled = DenseLayer(features, config.TRAIN.FC, act = act, name='downsample')
        
        # true selecting
        # 1000
        selecting_logits = DenseLayer(sampled, config.TRAIN.TIME_STEPS, act = None, name='selector')
        selecting = tl.layers.LambdaLayer(selecting_logits, fn = act, name='Selecting_softmax')
        #print(selecting.outputs.shape)
        selecting = tl.layers.ExpandDimsLayer(selecting, 2)
        # broadcasting to all embeded dimension
        selecting = TileLayer(selecting, [1,1,config.TRAIN.EMBED_DIM])
        # by visualizing selecting vector, can detect difference between species.
        return selecting, selecting_logits

def SelectorCNN(t_sequences, reuse = False):
    '''
    This parts plays an role as a selector of fixed position in genes, works like attention mechanism
    sequences: tf.place_holder([None, steps, embedding_dim])
    '''
    w_init = tf.random_normal_initializer(stddev=stddev)
    b_init = None
    g_init = tf.random_normal_initializer(1., stddev)
    act = lambda x: tf.nn.leaky_relu(x, 0.2)

    with tf.variable_scope("selectorCNN", reuse=reuse) as vs:
        # tl.layers.set_name_reuse(reuse) # remove for TL 1.8.0+
        sequences = InputLayer(t_sequences, name='in')
        # is it ok to add a embedding layer here
        # use strided convolution to decrease length of sequences
        #  
        #sequences = Conv1d(sequences, 32,KERNEL_SIZE , stride = 2, dilation_rate = 1, act = act, name = 'conv_500') # 500
        
        #sequences = Conv1d(sequences, 32, KERNEL_SIZE, stride = 2, dilation_rate = 1, act = act, name = 'conv_250') # 250 
        
        #sequences = Conv1d(sequences, 32, KERNEL_SIZE, stride = 2, dilation_rate = 1, act = act, name = 'conv_125') # 125
        
        #sequences = Conv1d(sequences, 32, KERNEL_SIZE, stride = 4, dilation_rate = 1, act = act, name = 'conv_63') # 125
        
        #sequences = Conv1d(sequences, 32, KERNEL_SIZE, stride = 4, dilation_rate = 1, act = act, name = 'conv_31') # 125
        
        #features = Conv1d(selected, 32, KERNEL_SIZE, stride = 1, dilation_rate = 1, act = act, name = 'conv1')
        
        features = Conv1d(sequences, 64, KERNEL_SIZE, stride = 2, act = act, name = 'conv1_stride')
        
        features = Conv1d(features, 64, KERNEL_SIZE, stride = 1, dilation_rate = 2, act = act, name = 'conv2')
        features = Conv1d(features, 128, KERNEL_SIZE, stride = 2, act = act, name = 'conv2_stride')
        
        features = Conv1d(features, 128, KERNEL_SIZE, stride = 1, dilation_rate = 4, act = act, name = 'conv3')
        features = Conv1d(features, 256, KERNEL_SIZE, stride = 2, act = act, name = 'conv3_stride')
        # stacking 3 bi-directiona,l lstm here
        
        #bi = BiRNNLayer(sequences, cell_fn = tf.contrib.rnn.LSTMCell, n_hidden = config.TRAIN.RNN_HIDDEN, n_steps = config.TRAIN.RNN_STEPS + 1, return_last = False, name = 'bi1')
        #bi = PReluLayer(bi, channel_shared = True, name='prelu1')
        
        #bi = BiRNNLayer(bi, cell_fn = tf.contrib.rnn.LSTMCell, n_hidden = config.TRAIN.RNN_HIDDEN, n_steps = config.TRAIN.TIME_STEPS, return_last = False, name='bi2')
        #bi = PReluLayer(bi, channel_shared = True, name = 'prelu2')
        
        #bi = BiRNNLayer(bi, cell_fn = tf.contrib.rnn.LSTMCell, n_hidden = config.TRAIN.RNN_HIDDEN, n_steps = config.TRAIN.RNN_STEPS + 1, return_last = False, name = 'bi3')
        #bi = PReluLayer(bi, channel_shared = True, name='prelu3')
        # use last outputs of bi-lstm to generate attention
        
        features = FlattenLayer(features, name='flatten_feature')
        
        # downsample was introduced for the overfitting issue
        sampled = DenseLayer(features, config.TRAIN.FC, act = act, name='downsample')
        
        # true selecting
        # 1000
        selecting_logits = DenseLayer(sampled, config.TRAIN.TIME_STEPS, act = tf.nn.softmax, name='selector')
        selecting = tl.layers.LambdaLayer(selecting_logits, fn = tf.nn.softmax, name='selector_softmax') 
        #print(selecting.outputs.shape)
        selecting = tl.layers.ExpandDimsLayer(selecting, 2)
        # broadcasting to all embeded dimension
        selecting = TileLayer(selecting, [1,1,config.TRAIN.EMBED_DIM])
        # by visualizing selecting vector, can detect difference between species.
        return selecting, selecting_logits
        
def Predictor(selecting, t_sequences, reuse = False):
    '''
    use seleceted features to do prediction
    '''
    
    w_init = tf.random_normal_initializer(stddev=stddev)
    b_init = None
    g_init = tf.random_normal_initializer(1., stddev)
    act = lambda x: tf.nn.leaky_relu(x, 0.2)
    
    with tf.variable_scope("predictor", reuse=tf.AUTO_REUSE) as vs:
        # tl.layers.set_name_reuse(reuse) # remove for TL 1.8.0+
        sequences = InputLayer(t_sequences, name='in')
        def scale(x):
            return 1000 * x
        selected = sequences
        #selecting = LambdaLayer(selecting, fn = scale, name='scale')
        #selected = ElementwiseLayer([selecting, sequences], combine_fn =  tf.multiply, name='selection')
                
        # USE convolution for computing? why?
        # use dialated convolution for larger reception field.
        # binding codon is 3
        
        # add depth for feature extraction
        pre = Conv1d(selected, 32, act = act, name = 'conv0')
        selected = pre
         
        for i in range(config.TRAIN.STACK_DEPTH):
            features = Conv1d(selected, 32, act = act, name = 'conv1_%d'%i)
            features = Conv1d(features, 32, act = None, name = 'conv2_%d'%i)
            selected = ElementwiseLayer([selected, features], combine_fn = tf.math.add, name = 'bypass_%d'%i)
        selected = ElementwiseLayer([pre, selected], combine_fn = tf.math.add, name = 'bypass_%d'%i)
        
        # google deepwave radio sl
        # downsample pooling dialation
        # no lstm, but larger reception field
        features = Conv1d(selected, 32, KERNEL_SIZE, stride = 1, dilation_rate = 1, act = act, name = 'conv1')
        
        features = Conv1d(selected, 64, KERNEL_SIZE, stride = 2, act = act, name = 'conv1_stride')
        
        features = Conv1d(features, 64, KERNEL_SIZE, stride = 1, dilation_rate = 2, act = act, name = 'conv2')
        features = Conv1d(features, 128, KERNEL_SIZE, stride = 2, act = act, name = 'conv2_stride')
        
        features = Conv1d(features, 128, KERNEL_SIZE, stride = 1, dilation_rate = 4, act = act, name = 'conv3')
        features = Conv1d(features, 256, KERNEL_SIZE, stride = 2, act = act, name = 'conv3_stride')
        
        
        features = FlattenLayer(features, name='flatten_features')
        
        hidden = DenseLayer(features, config.TRAIN.FC, name='hidden')
        
        hidden = PReluLayer(hidden, channel_shared = True, name='prelu1')
            
        category = DenseLayer(hidden, config.TRAIN.CLASSES, act = None, name = 'predicting')
        
         
        return category, tf.nn.softmax(category.outputs)

def sharedFeatureExtractor(t_sequences, name, reuse = False, is_train = True):

    w_init = tf.random_normal_initializer(stddev=stddev)
    b_init = None
    g_init = tf.random_normal_initializer(1., stddev)
    act = lambda x: tf.nn.leaky_relu(x, 0.2)
    
    with tf.variable_scope(name, reuse=reuse) as vs:

        sequences = InputLayer(t_sequences, name='in')
        #return sequences, sequences.outputs
        #return sequences        
        # user larger kernel size for the first layer

        feature1 = Conv1d(sequences, 300, 20, stride = 1, dilation_rate = 1, act = None, name = 'conv_500') # 500
        feature1 = tl.layers.BatchNormLayer(feature1, beta_init = w_init, gamma_init = w_init, is_train = is_train, name='bn1')
        feature1 = PReluLayer(feature1, channel_shared = True, name='conv1_relu')
        if config.TRAIN.DROPOUT:
            feature1 = DropoutLayer(feature1, keep = config.TRAIN.DROPOUT_KEEP, name = 'drop_features1', is_fix = True)
        
        feature1 = SelfAttentionLayer(feature1, 8, 32, name='attention1')
        # used to simulate gapped kmer
        #feature2 = Conv1d(sequences, 300, 20, stride = 1, dilation_rate = 2, act = None, name = 'conv_8_2') # 500
        #features = tl.layers.BatchNormLayer(features, beta_init = w_init, gamma_init = w_init, is_train = is_train, name='bn1')
        #feature2 = PReluLayer(feature2, channel_shared = True, name='conv1_2_relu')
        
        #feature3 = Conv1d(sequences, 300, 20, stride = 1, dilation_rate = 4, act = None, name = 'conv_16_2') # 500
        #features = tl.layers.BatchNormLayer(features, beta_init = w_init, gamma_init = w_init, is_train = is_train, name='bn1')
        #feature3 = PReluLayer(feature3, channel_shared = True, name='conv1_3_relu')
        
        #features = ConcatLayer([feature1, feature2,  feature3], name = 'concat')
        
        features = Conv1d(feature1, 32, KERNEL_SIZE, stride = 1, dilation_rate = 1, act = None, name = 'conva_250') # 250
        features = tl.layers.BatchNormLayer(features, beta_init = w_init, gamma_init = w_init, is_train = is_train, name='bna2')
        features = PReluLayer(features, channel_shared = True, name='conv2a_relu')
        if config.TRAIN.DROPOUT:
            features = DropoutLayer(features, keep = config.TRAIN.DROPOUT_KEEP, name = 'drop_features', is_fix = True)

        
        features = Conv1d(features, 32, KERNEL_SIZE, stride = 1, dilation_rate = 1, act = None, name = 'conv_250') # 250
        features = tl.layers.BatchNormLayer(features, beta_init = w_init, gamma_init = w_init, is_train = is_train, name='bn2')
        features = PReluLayer(features, channel_shared = True, name='conv2_relu')
        if config.TRAIN.DROPOUT:
            features = DropoutLayer(features, keep = config.TRAIN.DROPOUT_KEEP, name = 'drop_features_2', is_fix = True)

        
        features = Conv1d(features, 32, KERNEL_SIZE, stride = 1, dilation_rate = 1, act = None, name = 'conv_125') # 125
        features = tl.layers.BatchNormLayer(features, beta_init = w_init, gamma_init = w_init, is_train = is_train, name='bn3')
        features = PReluLayer(features, channel_shared = True, name='conv3_relu')
        if config.TRAIN.DROPOUT:
            features = DropoutLayer(features, keep = config.TRAIN.DROPOUT_KEEP, name = 'drop_features_3', is_fix = True)
        #sequences = Conv1d(sequences, 32, KERNEL_SIZE, stride = 4, dilation_rate = 1, act = act, name = 'conv_63') # 125
        
        #sequences = Conv1d(sequences, 32, KERNEL_SIZE, stride = 4, dilation_rate = 1, act = act, name = 'conv_31') # 125
        
        
        # stacking 3 bi-directiona,l lstm here
        
        features = BiRNNLayer(features, cell_fn = tf.contrib.rnn.LSTMCell, n_hidden = config.TRAIN.RNN_HIDDEN, n_steps = config.TRAIN.RNN_STEPS + 1, return_last = False, name = 'bi1')
        #features = PReluLayer(features, channel_shared = True, name='prelu1')
        
        #
        '''
        features = Conv1d(sequences, 32, KERNEL_SIZE, stride = 2, dilation_rate = 1, act = None, name = 'conv1')
                
        features = tl.layers.BatchNormLayer(features, beta_init = w_init, gamma_init = w_init, is_train = is_train, name='bn1')
        
        features = PReluLayer(features, channel_shared = True, name='conv1_relu')

        features = Conv1d(features, 64, KERNEL_SIZE, stride = 2, act = None, name = 'conv1_stride')
        
        features = tl.layers.BatchNormLayer(features, beta_init = w_init, gamma_init = w_init, is_train = is_train, name='bn2')

        features = PReluLayer(features, channel_shared = True, name='conv2_relu')

        features = Conv1d(features, 64, KERNEL_SIZE, stride = 2, act = None, name = 'conv2_stride')
        
        features = tl.layers.BatchNormLayer(features, beta_init = w_init, gamma_init = w_init, is_train = is_train, name='bn3')

        features = PReluLayer(features, channel_shared = True, name='conv3_relu')
        '''
        
        return features, feature1.outputs

def sharedFeatureExtractor2(t_sequences, name, reuse = False, is_train = True):
    
    w_init = tf.random_normal_initializer(stddev=stddev)
    b_init = None
    g_init = tf.random_normal_initializer(1., stddev)
    act = lambda x: tf.nn.leaky_relu(x, 0.2)
    
    kernels = config.TRAIN.KERNEL.split('_')

    with tf.variable_scope(name, reuse=reuse) as vs:

        sequences = InputLayer(t_sequences, name='in')
        #return sequences, sequences.outputs
        #return sequences        
        # user larger kernel size for the first layer
        feature_conv = Conv1d(sequences, 300, int(kernels[0]), stride = 1, dilation_rate = 1, act = None, name = 'conv_500') # 500
        feature1 = tl.layers.BatchNormLayer(feature_conv, beta_init = w_init, gamma_init = w_init, is_train = is_train, name='bn1')
        feature1 = PReluLayer(feature1, channel_shared = True, name='conv1_relu')
        if config.TRAIN.DROPOUT:
            feature1 = DropoutLayer(feature1, keep = config.TRAIN.DROPOUT_KEEP, name = 'drop_features1', is_fix = True, is_train = is_train)
        
        # used to simulate gapped kmer
        feature2 = Conv1d(sequences, 300, int(kernels[1]), stride = 1, dilation_rate = 2, act = None, name = 'conv_8_2') # 500
        feature2 = tl.layers.BatchNormLayer(feature2, beta_init = w_init, gamma_init = w_init, is_train = is_train, name='feature2_bn')
        feature2 = PReluLayer(feature2, channel_shared = True, name='conv1_2_relu')
        if config.TRAIN.DROPOUT:
            feature2 = DropoutLayer(feature2, keep = config.TRAIN.DROPOUT_KEEP, name = 'drop_features2', is_fix = True, is_train = is_train)


        feature3 = Conv1d(sequences, 300, int(kernels[2]), stride = 1, dilation_rate = 4, act = None, name = 'conv_16_2') # 500
        feature3 = tl.layers.BatchNormLayer(feature3, beta_init = w_init, gamma_init = w_init, is_train = is_train, name='bn2')
        feature3 = PReluLayer(feature3, channel_shared = True, name='conv1_3_relu')
        if config.TRAIN.DROPOUT:
            feature3 = DropoutLayer(feature3, keep = config.TRAIN.DROPOUT_KEEP, name = 'drop_features3', is_fix = True, is_train = is_train)

        features = ConcatLayer([feature1, feature2,  feature3], name = 'concat')
        
        
        features = Conv1d(features, 32, KERNEL_SIZE, stride = 1, dilation_rate = 1, act = None, name = 'conva_250') # 250
        features = tl.layers.BatchNormLayer(features, beta_init = w_init, gamma_init = w_init, is_train = is_train, name='bna3')
        con_features = PReluLayer(features, channel_shared = True, name='conv2a_relu')
        if config.TRAIN.DROPOUT:
            con_features = DropoutLayer(con_features, keep = config.TRAIN.DROPOUT_KEEP, name = 'drop_features4', is_fix = True, is_train = is_train)        

        
        features = Conv1d(con_features, 32, KERNEL_SIZE, stride = 1, dilation_rate = 1, act = None, name = 'conva_250_c') # 250
        features = tl.layers.BatchNormLayer(features, beta_init = w_init, gamma_init = w_init, is_train = is_train, name='bna3_c')
        features = PReluLayer(features, channel_shared = True, name='conv2a_relu_c')
        if config.TRAIN.DROPOUT:
            features = DropoutLayer(features, keep = config.TRAIN.DROPOUT_KEEP, name = 'drop_featuress1', is_fix = True, is_train = is_train) 

        features = Conv1d(features, 32, KERNEL_SIZE, stride = 1, dilation_rate = 1, act = None, name = 'conv_250') # 250
        features = tl.layers.BatchNormLayer(features, beta_init = w_init, gamma_init = w_init, is_train = is_train, name='bn4')
        features = PReluLayer(features, channel_shared = True, name='conv2_relu')
        if config.TRAIN.DROPOUT:
            features = DropoutLayer(features, keep = config.TRAIN.DROPOUT_KEEP, name = 'drop_featuresss2', is_fix = True, is_train = is_train) 

        features = ElementwiseLayer([features, con_features], tf.add, name = 'elem_add')
        
        features = Conv1d(features, 32, KERNEL_SIZE, stride = 1, dilation_rate = 1, act = None, name = 'conv_125') # 125
        features = tl.layers.BatchNormLayer(features, beta_init = w_init, gamma_init = w_init, is_train = is_train, name='bn5')
        features = PReluLayer(features, channel_shared = True, name='conv3_relu')
        if config.TRAIN.DROPOUT:
            features = DropoutLayer(features, keep = config.TRAIN.DROPOUT_KEEP, name = 'drop_featuresss3', is_fix = True, is_train = is_train) 
        #sequences = Conv1d(sequences, 32, KERNEL_SIZE, stride = 4, dilation_rate = 1, act = act, name = 'conv_63') # 125
        
        #sequences = Conv1d(sequences, 32, KERNEL_SIZE, stride = 4, dilation_rate = 1, act = act, name = 'conv_31') # 125
        
        
        # stacking 3 bi-directiona,l lstm here
        
        features = BiRNNLayer(features, cell_fn = tf.contrib.rnn.LSTMCell, n_hidden = config.TRAIN.RNN_HIDDEN, n_steps = config.TRAIN.RNN_STEPS + 1, return_last = False, name = 'bi1')
        #features = PReluLayer(features, channel_shared = True, name='prelu1')
        #features = BiRNNLayer(features, cell_fn = tf.contrib.rnn.LSTMCell, n_hidden = config.TRAIN.RNN_HIDDEN, n_steps = config.TRAIN.RNN_STEPS + 1, return_last = False, name = 'bi2')
        #
        features = SelfAttentionLayer(features, 8 , 128,name='self-attention')
        features = SelfAttentionLayer(features, 8 , 128,name='self-attention2')
        '''

        features = Conv1d(sequences, 32, KERNEL_SIZE, stride = 2, dilation_rate = 1, act = None, name = 'conv1')
                
        features = tl.layers.BatchNormLayer(features, beta_init = w_init, gamma_init = w_init, is_train = is_train, name='bn1')
        
        features = PReluLayer(features, channel_shared = True, name='conv1_relu')

        features = Conv1d(features, 64, KERNEL_SIZE, stride = 2, act = None, name = 'conv1_stride')
        
        features = tl.layers.BatchNormLayer(features, beta_init = w_init, gamma_init = w_init, is_train = is_train, name='bn2')

        features = PReluLayer(features, channel_shared = True, name='conv2_relu')

        features = Conv1d(features, 64, KERNEL_SIZE, stride = 2, act = None, name = 'conv2_stride')
        
        features = tl.layers.BatchNormLayer(features, beta_init = w_init, gamma_init = w_init, is_train = is_train, name='bn3')

        features = PReluLayer(features, channel_shared = True, name='conv3_relu')
        '''
        
        return features, feature_conv.outputs#, features.outputs


def attention(feature, name):
    hidden = tl.layers.TimeDistributedLayer(feature, layer_class=tl.layers.DenseLayer, args={'n_units':64, 'name':name + 'dense','act' :tf.nn.tanh}, name= name + 'time_dense')
    hidden = tl.layers.TimeDistributedLayer(hidden, layer_class=tl.layers.DenseLayer, args={'n_units':1, 'name':name + 'dense2'}, name= name + 'time_dense2')
    hidden = tl.layers.FlattenLayer(hidden, name = name + 'flatten')
    return LambdaLayer(hidden, fn = tf.nn.softmax, name = name + "_softmax")

def sharedFeatureExtractor2D(t_sequences, name, reuse = False, is_train=True):

    w_init = tf.random_normal_initializer(stddev=stddev)
    b_init = None
    g_init = tf.random_normal_initializer(1., stddev)
    act = lambda x: tf.nn.leaky_relu(x, 0.2)
    
    with tf.variable_scope(name, reuse=reuse) as vs:

        sequences = InputLayer(t_sequences, name='in')
        #return sequences        
        features = Conv2d(sequences, 32,KERNEL_SIZE , stride = 2, dilation_rate = 1, act = None, name = 'conv_500') # 500
        #features = tl.layers.BatchNormLayer(features, beta_init = w_init, gamma_init = w_init, is_train = is_train, name='bn1')
        features = PReluLayer(features, channel_shared = True, name='conv1_relu')
        
        features = Conv2d(features, 32, KERNEL_SIZE, stride = 2, dilation_rate = 1, act = None, name = 'conv_250') # 250
        #features = tl.layers.BatchNormLayer(features, beta_init = w_init, gamma_init = w_init, is_train = is_train, name='bn2')
        features = PReluLayer(features, channel_shared = True, name='conv2_relu')
        
        
        features = Conv2d(features, 32, KERNEL_SIZE, stride = 2, dilation_rate = 1, act = None, name = 'conv_125') # 125
        #features = tl.layers.BatchNormLayer(features, beta_init = w_init, gamma_init = w_init, is_train = is_train, name='bn3')
        features = PReluLayer(features, channel_shared = True, name='conv3_relu')
        

        #sequences = Conv1d(sequences, 32, KERNEL_SIZE, stride = 4, dilation_rate = 1, act = act, name = 'conv_63') # 125
        
        #sequences = Conv1d(sequences, 32, KERNEL_SIZE, stride = 4, dilation_rate = 1, act = act, name = 'conv_31') # 125
        ''''
        features_ex = Conv1d(features, 32, KERNEL_SIZE, act = None, name = 'conv_same') # 125
        features_ex = tl.layers.BatchNormLayer(features_ex, beta_init = w_init, gamma_init = w_init, is_train = is_train, name='bn_same')
        features_ex = PReluLayer(features_ex, channel_shared = True, name='convsame_relu')
        # Introducing self-attention here
        attention_map = AttentionLayer(features, name = 'Extractor_')
        #attention_map = attention(features, 'Extractor_')
        attention_map = tl.layers.ExpandDimsLayer(attention_map, 2)
        attention_map = TileLayer(attention_map, [1,1,32])
        features_masked = ElementwiseLayer([attention_map, features], combine_fn =  tf.multiply, name='selection')
        # different species will have different attention
        features = tl.layers.ConcatLayer([features_ex, features_masked], -1, name ='concat_layer')
        # stacking 3 bi-directiona,l lstm here
        '''
        features = BiRNNLayer(features, cell_fn = tf.contrib.rnn.LSTMCell, n_hidden = int(config.TRAIN.RNN_HIDDEN/4), n_steps = config.TRAIN.RNN_STEPS + 1, return_last = False, name = 'bi1')
        #features = PReluLayer(features, channel_shared = True, name='prelu1')
        #self-attention mechanism       
        '''
        features = Conv1d(sequences, 32, KERNEL_SIZE, stride = 2, dilation_rate = 1, act = None, name = 'conv1')
                
        features = tl.layers.BatchNormLayer(features, beta_init = w_init, gamma_init = w_init, is_train = is_train, name='bn1')
        
        features = PReluLayer(features, channel_shared = True, name='conv1_relu')

        features = Conv1d(features, 64, KERNEL_SIZE, stride = 2, act = None, name = 'conv1_stride')
        
        features = tl.layers.BatchNormLayer(features, beta_init = w_init, gamma_init = w_init, is_train = is_train, name='bn2')

        features = PReluLayer(features, channel_shared = True, name='conv2_relu')

        features = Conv1d(features, 64, KERNEL_SIZE, stride = 2, act = None, name = 'conv2_stride')
        
        features = tl.layers.BatchNormLayer(features, beta_init = w_init, gamma_init = w_init, is_train = is_train, name='bn3')

        features = PReluLayer(features, channel_shared = True, name='conv3_relu')
        '''
        return features
 
def classifier(features, name, reuse = False, is_train = True):

    w_init = tf.random_normal_initializer(stddev=stddev)
    b_init = None 
    g_init = tf.random_normal_initializer(1., stddev)
    act = lambda x: tf.nn.leaky_relu(x, 0.2)
    
    with tf.variable_scope(name, reuse=reuse) as vs:

        conv_features = Conv1d(features, 32, KERNEL_SIZE, stride = 1, dilation_rate = 1, act = None, name = 'conv1')
        features = tl.layers.BatchNormLayer(conv_features, beta_init = w_init, gamma_init = w_init, is_train = is_train, name='bn1')
        features = PReluLayer(features, channel_shared = True, name='conv1_relu')
        #if config.TRAIN.DROPOUT:
        #    features = DropoutLayer(features, keep = config.TRAIN.DROPOUT_KEEP, name = 'drop_features_1', is_fix = True, is_train = is_train)

        #features = ConcatLayer([features, seq_features], name = 'seq_concat')

        features = Conv1d(features, 64, KERNEL_SIZE, stride = 1, act = None, name = 'conv1_stride')
        features = tl.layers.BatchNormLayer(features, beta_init = w_init, gamma_init = w_init, is_train = is_train, name='bn2')
        fin_features = PReluLayer(features, channel_shared = True, name='conv2_relu')
        #if config.TRAIN.DROPOUT:
        #    features = DropoutLayer(features, keep = config.TRAIN.DROPOUT_KEEP, name = 'drop_features_2', is_fix = True, is_train = is_train)

        features = FlattenLayer(fin_features, name='flatten_features')

        features = DenseLayer(features, config.TRAIN.FC, act = None, name='hidden')
        features = tl.layers.BatchNormLayer(features, beta_init = w_init, gamma_init = w_init, is_train = is_train, name='bn3') 
        hidden = PReluLayer(features, channel_shared = True, name='prelu1')
        if config.TRAIN.DROPOUT:
            features = DropoutLayer(features, keep = config.TRAIN.DROPOUT_KEEP, name = 'drop_features_3', is_fix = True, is_train = is_train)

        category = DenseLayer(hidden, 2, act = None, name = 'predicting')
        if config.TRAIN.DROPOUT:
            features = DropoutLayer(features, keep = config.TRAIN.DROPOUT_KEEP, name = 'drop_features_3', is_fix = True, is_train = is_train)
        return category#, conv_features

def classifierSequences(features, t_sequences, name, reuse, is_train):

    w_init = tf.random_normal_initializer(stddev=stddev)
    b_init = None
    g_init = tf.random_normal_initializer(1., stddev)
    act = lambda x: tf.nn.leaky_relu(x, 0.2)
    
    with tf.variable_scope(name, reuse=reuse) as vs:

        sequences = InputLayer(t_sequences, name='in')

        seq_features = Conv1d(sequences, 32, KERNEL_SIZE, stride = 1, dilation_rate = 1, act = None, name = 'seq_conv1')
        seq_features = tl.layers.BatchNormLayer(seq_features, beta_init = w_init, gamma_init = w_init, is_train = is_train, name='seq_bn1')
        seq_features = PReluLayer(seq_features, channel_shared = True, name='seq_conv1_relu')


        seq_features1 = Conv1d(seq_features, 32, KERNEL_SIZE, stride = 1, dilation_rate = 1, act = None, name = 'res_seq_conv1')
        #seq_features = tl.layers.BatchNormLayer(seq_features, beta_init = w_init, gamma_init = w_init, is_train = is_train, name='seq_bn1')
        seq_features1 = PReluLayer(seq_features1, channel_shared = True, name='res_seq_conv1_relu')
        seq_features1 = Conv1d(seq_features1, 32, KERNEL_SIZE, stride = 1, dilation_rate = 1, act = None, name = 'res_seq_conv1')


        
        seq_features = ElementwiseLayer([seq_features, seq_features1], tf.add, name = 'elem_add')

        seq_features = SelfAttentionLayer(seq_features, 8,128,name='seq_attention')

        seq_features = Conv1d(seq_features, 32, KERNEL_SIZE, stride = 1, dilation_rate = 1, act = None, name = '_seq_conv1')
        seq_features = tl.layers.BatchNormLayer(seq_features, beta_init = w_init, gamma_init = w_init, is_train = is_train, name='_seq_bn1')
        seq_features = PReluLayer(seq_features, channel_shared = True, name='_res_seq_conv1_relu')
        '''
        if config.TRAIN.DROPOUT:
            seq_features = DropoutLayer(seq_features, keep = config.TRAIN.DROPOUT_KEEP, name = 'seq_drop_features_1', is_fix = True, is_train = is_train)
        '''
        
        features = Conv1d(features, 32, KERNEL_SIZE, stride = 1, dilation_rate = 1, act = None, name = 'conv1')
        features = tl.layers.BatchNormLayer(features, beta_init = w_init, gamma_init = w_init, is_train = is_train, name='bn1')
        features = PReluLayer(features, channel_shared = True, name='conv1_relu')
        '''
        if config.TRAIN.DROPOUT:
            features = DropoutLayer(features, DROPOUT_KEEP = config.TRAIN.DROPOUT_KEEP, name = 'drop_features_1', is_fix = True, is_train = is_train)
        '''
        features = ConcatLayer([features, seq_features], name = 'seq_concat')

        features = Conv1d(features, 64, KERNEL_SIZE, stride = 1, act = None, name = 'conv1_stride')
        features = tl.layers.BatchNormLayer(features, beta_init = w_init, gamma_init = w_init, is_train = is_train, name='bn2')
        features = PReluLayer(features, channel_shared = True, name='conv2_relu')
        '''
        if config.TRAIN.DROPOUT:
            features = DropoutLayer(features, keep = config.TRAIN.DROPOUT_KEEP, name = 'drop_features_2', is_fix = True, is_train = is_train)
        '''
        features = FlattenLayer(features, name='flatten_features')

        features = DenseLayer(features, config.TRAIN.FC, act = None, name='hidden')
        features = tl.layers.BatchNormLayer(features, beta_init = w_init, gamma_init = w_init, is_train = is_train, name='bn3') 
        hidden = PReluLayer(features, channel_shared = True, name='prelu1')
        if config.TRAIN.DROPOUT:
            features = DropoutLayer(features, keep = config.TRAIN.DROPOUT_KEEP, name = 'drop_features_3', is_fix = True, is_train = is_train)

        category = DenseLayer(hidden, 2, act = None, name = 'predicting')
        if config.TRAIN.DROPOUT:
            features = DropoutLayer(features, keep = config.TRAIN.DROPOUT_KEEP, name = 'drop_featudres_3', is_fix = True, is_train = is_train)
        return category

def DeepM6ASeq_pre(t_sequences, name, reuse = False, is_train = True):
    
    w_init = tf.random_normal_initializer(stddev=stddev)
    b_init = None
    g_init = tf.random_normal_initializer(1., stddev)
    act = lambda x: tf.nn.leaky_relu(x, 0.2)
    
    kernels = config.TRAIN.KERNEL.split('_')

    with tf.variable_scope(name, reuse=reuse) as vs:

        sequences = InputLayer(t_sequences, name='in')
        return sequences, sequences.outputs

def DeepM6ASeq(features, name, reuse = False, is_train = True):
    w_init = tf.random_normal_initializer(stddev=stddev)
    b_init = None 
    g_init = tf.random_normal_initializer(1., stddev)
    act = lambda x: tf.nn.leaky_relu(x, 0.2)
    
    with tf.variable_scope(name, reuse=reuse) as vs:

        
        features = Conv1d(features, 256, 10, stride = 1, dilation_rate = 1, act = None, name = 'conv1')
        #MaxPool1d(features,)
        features = tl.layers.BatchNormLayer(features, beta_init = w_init, gamma_init = w_init, is_train = is_train, name='bn1')
        features = PReluLayer(features, channel_shared = True, name='conv1_relu')
        if config.TRAIN.DROPOUT:
            features = DropoutLayer(features, keep = 0.5, name = 'drop_features_1', is_fix = True, is_train = is_train)

        features = Conv1d(features, 64, 5, stride = 1, act = None, name = 'conv1_stride')
        features = tl.layers.BatchNormLayer(features, beta_init = w_init, gamma_init = w_init, is_train = is_train, name='bn2')
        features = PReluLayer(features, channel_shared = True, name='conv2_relu')
        if config.TRAIN.DROPOUT:
            features = DropoutLayer(features, keep = 0.5, name = 'drop_features_2', is_fix = True, is_train = is_train)


        fin_features = BiRNNLayer(features, cell_fn = tf.contrib.rnn.LSTMCell, n_hidden = 32, n_steps = config.TRAIN.RNN_STEPS + 1, return_last = False, name = 'bi1')
        #MaxPool1d


        features = FlattenLayer(fin_features, name='flatten_features')

        #features = DenseLayer(features, config.TRAIN.FC, act = None, name='hidden')
        #features = tl.layers.BatchNormLayer(features, beta_init = w_init, gamma_init = w_init, is_train = is_train, name='bn3') 
        #hidden = PReluLayer(features, channel_shared = True, name='prelu1')
        if config.TRAIN.DROPOUT:
            features = DropoutLayer(features, keep = config.TRAIN.DROPOUT_KEEP, name = 'drop_features_3', is_fix = True, is_train = is_train)

        category = DenseLayer(features, 2, act = None, name = 'predicting')
        return category, fin_features
 
def sharedFeatureExtractor3(t_sequences, name, reuse = False, is_train = True):
    
    '''
    Use attention to replace the LSTM layer
    '''
    w_init = tf.random_normal_initializer(stddev=0.2)
    b_init = None
    g_init = tf.random_normal_initializer(1., 0.2)
    act = lambda x: tf.nn.leaky_relu(x, 0.2)
    
    kernels = config.TRAIN.KERNEL.split('_')

    with tf.variable_scope(name, reuse=reuse) as vs:

        sequences = InputLayer(t_sequences, name='in')
        embedding = EmbeddingInputlayer(sequences, 5, 32)
        #return sequences, sequences.outputs
        #return sequences        
        # user larger kernel size for the first layer
        feature1 = Conv1d(embedding, 300, int(kernels[0]), stride = 1, dilation_rate = 1, act = None, name = 'conv_500') # 500
        feature1 = tl.layers.BatchNormLayer(feature1, beta_init = w_init, gamma_init = w_init, is_train = is_train, name='bn1')
        feature1 = PReluLayer(feature1, channel_shared = True, name='conv1_relu')
        '''
        if config.TRAIN.DROPOUT:
            feature1 = DropoutLayer(feature1, keep = config.TRAIN.DROPOUT_KEEP, name = 'drop_features1', is_fix = True)
        '''

        # used to simulate gapped kmer
        feature2 = Conv1d(embedding, 300, int(kernels[1]), stride = 1, dilation_rate = 2, act = None, name = 'conv_8_2') # 500
        feature2 = tl.layers.BatchNormLayer(feature2, beta_init = w_init, gamma_init = w_init, is_train = is_train, name='feature2_bn')
        feature2 = PReluLayer(feature2, channel_shared = True, name='conv1_2_relu')
        '''
        if config.TRAIN.DROPOUT:
            feature2 = DropoutLayer(feature2, keep = config.TRAIN.DROPOUT_KEEP, name = 'drop_features2', is_fix = True)
        '''

        feature3 = Conv1d(embedding, 300, int(kernels[2]), stride = 1, dilation_rate = 4, act = None, name = 'conv_16_2') # 500
        feature3 = tl.layers.BatchNormLayer(feature3, beta_init = w_init, gamma_init = w_init, is_train = is_train, name='bn2')
        feature3 = PReluLayer(feature3, channel_shared = True, name='conv1_3_relu')
        '''
        if config.TRAIN.DROPOUT:
            feature3 = DropoutLayer(feature3, keep = config.TRAIN.DROPOUT_KEEP, name = 'drop_features3', is_fix = True)
        '''
        features = ConcatLayer([feature1, feature2,  feature3], name = 'concat')
        
        
        features = Conv1d(features, 32, KERNEL_SIZE, stride = 1, dilation_rate = 1, act = None, name = 'conva_250') # 250
        features = tl.layers.BatchNormLayer(features, beta_init = w_init, gamma_init = w_init, is_train = is_train, name='bna3')
        con_features = PReluLayer(features, channel_shared = True, name='conv2a_relu')
        '''
        if config.TRAIN.DROPOUT:
            con_features = DropoutLayer(con_features, keep = config.TRAIN.DROPOUT_KEEP, name = 'drop_features4', is_fix = True)        
        '''
        
        features = Conv1d(con_features, 32, KERNEL_SIZE, stride = 1, dilation_rate = 1, act = None, name = 'conva_250_c') # 250
        features = tl.layers.BatchNormLayer(features, beta_init = w_init, gamma_init = w_init, is_train = is_train, name='bna3_c')
        features = PReluLayer(features, channel_shared = True, name='conv2a_relu_c')
        '''
        if config.TRAIN.DROPOUT:
            features = DropoutLayer(features, keep = config.TRAIN.DROPOUT_KEEP, name = 'drop_featuress1', is_fix = True) 
        '''
        features = Conv1d(features, 32, KERNEL_SIZE, stride = 1, dilation_rate = 1, act = None, name = 'conv_250') # 250
        features = tl.layers.BatchNormLayer(features, beta_init = w_init, gamma_init = w_init, is_train = is_train, name='bn4')
        features = PReluLayer(features, channel_shared = True, name='conv2_relu')
        '''
        if config.TRAIN.DROPOUT:
            features = DropoutLayer(features, keep = config.TRAIN.DROPOUT_KEEP, name = 'drop_featuresss2', is_fix = True) 
        '''
        features = ElementwiseLayer([features, con_features], tf.add, name = 'elem_add')
        
        features = Conv1d(features, 32, KERNEL_SIZE, stride = 1, dilation_rate = 1, act = None, name = 'conv_125') # 125
        features = tl.layers.BatchNormLayer(features, beta_init = w_init, gamma_init = w_init, is_train = is_train, name='bn5')
        features = PReluLayer(features, channel_shared = True, name='conv3_relu')
        '''
        if config.TRAIN.DROPOUT:
            features = DropoutLayer(features, keep = config.TRAIN.DROPOUT_KEEP, name = 'drop_featuresss3', is_fix = True) 
        '''
        #sequences = Conv1d(sequences, 32, KERNEL_SIZE, stride = 4, dilation_rate = 1, act = act, name = 'conv_63') # 125
        
        #sequences = Conv1d(sequences, 32, KERNEL_SIZE, stride = 4, dilation_rate = 1, act = act, name = 'conv_31') # 125
        
        
        # stacking 3 bi-directiona,l lstm here
        '''
        features = BiRNNLayer(features, cell_fn = tf.contrib.rnn.LSTMCell, n_hidden = config.TRAIN.RNN_HIDDEN, n_steps = config.TRAIN.RNN_STEPS + 1, return_last = False, name = 'bi1')
        features = SelfAttentionLayer(features, 8 , 128,name='self-attention')
        #features = PReluLayer(features, channel_shared = True, name='prelu1')
        #features = BiRNNLayer(features, cell_fn = tf.contrib.rnn.LSTMCell, n_hidden = config.TRAIN.RNN_HIDDEN, n_steps = config.TRAIN.RNN_STEPS + 1, return_last = False, name = 'bi2')
        #
        '''
        def my_rev(inputs):
            return tf.reverse(inputs, [1])
        rev_features = LambdaLayer(features, my_rev, name ='reverse')
        rev_features = SelfAttentionLayer(rev_features, 8 , 128,name='rev_self-attention')
        #rev_features = TimeDistributedLayer(rev_features, layer_class=tl.layers.DenseLayer, args={'n_units':50, 'name':'dense_rev'}, name='time_dense_rev')

        #DenseLayer(hidden, 2, act = None, name = 'predicting')


        features = SelfAttentionLayer(features, 8 , 128,name='self-attention')
        #rev_features = TimeDistributedLayer(rev_features, layer_class=tl.layers.DenseLayer, args={'n_units':50, 'name':'dense1'}, name='time_dense')

        features = ConcatLayer([features, rev_features], name = 'attention_concat')
        
        '''
        
        features = Conv1d(sequences, 32, KERNEL_SIZE, stride = 2, dilation_rate = 1, act = None, name = 'conv1')
                
        features = tl.layers.BatchNormLayer(features, beta_init = w_init, gamma_init = w_init, is_train = is_train, name='bn1')
        
        features = PReluLayer(features, channel_shared = True, name='conv1_relu')

        features = Conv1d(features, 64, KERNEL_SIZE, stride = 2, act = None, name = 'conv1_stride')
        
        features = tl.layers.BatchNormLayer(features, beta_init = w_init, gamma_init = w_init, is_train = is_train, name='bn2')

        features = PReluLayer(features, channel_shared = True, name='conv2_relu')

        features = Conv1d(features, 64, KERNEL_SIZE, stride = 2, act = None, name = 'conv2_stride')
        
        features = tl.layers.BatchNormLayer(features, beta_init = w_init, gamma_init = w_init, is_train = is_train, name='bn3')

        features = PReluLayer(features, channel_shared = True, name='conv3_relu')
        '''
        
        return features, feature1.outputs

def AttentionSeqs(t_sequences, name, is_train= True, reuse = False):

    with tf.variable_scope(name, reuse=reuse) as vs:
        sequences = InputLayer(t_sequences, name='in')

        embedding = EmbeddingInputlayer(sequences, 5, 32)

        def my_rev(inputs):
            return tf.reverse(inputs, [1])
        def pe(inputs):
            return Position_Embedding(inputs, 32)

        rev_features = LambdaLayer(embedding, my_rev, name ='reverse')
        rev_pos_embed = LambdaLayer(rev_features, pe, name='rev_position-embedding')
        rev_features = ConcatLayer([rev_features, rev_pos_embed], name = 'rev_embedding_concat')

        for i in range(6):

            rev_features = SelfAttentionLayer(rev_features, 8 , 128,name='rev_self-attention%d'%i)
            #rev_features = TimeDistributedLayer(rev_features, layer_class=tl.layers.DenseLayer, args={'n_units':50, 'name':'dense1'}, name='time_dense')
            rev_features = Conv1d(rev_features, 32, KERNEL_SIZE, stride = 1, dilation_rate = 1, act = tf.nn.relu, name = 'rev_conv_125_%d'%i)

        pos_embed = LambdaLayer(embedding, pe, name='position-embedding')
        features = ConcatLayer([pos_embed, embedding], name = 'embedding_concat')

        for i in range(6):

            features = SelfAttentionLayer(features, 8 , 128,name='self-attention%d'%i)
            #rev_features = TimeDistributedLayer(rev_features, layer_class=tl.layers.DenseLayer, args={'n_units':50, 'name':'dense1'}, name='time_dense')
            features = Conv1d(features, 32, KERNEL_SIZE, stride = 1, dilation_rate = 1, act = tf.nn.relu, name = 'conv_125_%d'%i)

        features = ConcatLayer([rev_features, features], name = 'embedding_concat')

        return features, features.outputs

def sharedFeatureExtractor_nodropout(t_sequences, name, reuse = False, is_train = True):
    
    w_init = tf.random_normal_initializer(stddev=stddev)
    b_init = None
    g_init = tf.random_normal_initializer(1., stddev)
    act = lambda x: tf.nn.leaky_relu(x, 0.2)
    
    kernels = config.TRAIN.KERNEL.split('_')

    with tf.variable_scope(name, reuse=reuse) as vs:

        sequences = InputLayer(t_sequences, name='in')
        #return sequences, sequences.outputs
        #return sequences        
        # user larger kernel size for the first layer
        feature1 = Conv1d(sequences, 300, int(kernels[0]), stride = 1, dilation_rate = 1, act = None, name = 'conv_500') # 500
        feature1 = tl.layers.BatchNormLayer(feature1, beta_init = w_init, gamma_init = w_init, is_train = is_train, name='bn1')
        feature1 = PReluLayer(feature1, channel_shared = True, name='conv1_relu')
        
        # used to simulate gapped kmer
        feature2 = Conv1d(sequences, 300, int(kernels[1]), stride = 1, dilation_rate = 2, act = None, name = 'conv_8_2') # 500
        feature2 = tl.layers.BatchNormLayer(feature2, beta_init = w_init, gamma_init = w_init, is_train = is_train, name='feature2_bn')
        feature2 = PReluLayer(feature2, channel_shared = True, name='conv1_2_relu')


        feature3 = Conv1d(sequences, 300, int(kernels[2]), stride = 1, dilation_rate = 4, act = None, name = 'conv_16_2') # 500
        feature3 = tl.layers.BatchNormLayer(feature3, beta_init = w_init, gamma_init = w_init, is_train = is_train, name='bn2')
        feature3 = PReluLayer(feature3, channel_shared = True, name='conv1_3_relu')

        features = ConcatLayer([feature1, feature2,  feature3], name = 'concat')
        
        
        features = Conv1d(features, 32, KERNEL_SIZE, stride = 1, dilation_rate = 1, act = None, name = 'conva_250') # 250
        features = tl.layers.BatchNormLayer(features, beta_init = w_init, gamma_init = w_init, is_train = is_train, name='bna3')
        con_features = PReluLayer(features, channel_shared = True, name='conv2a_relu')      

        
        features = Conv1d(con_features, 32, KERNEL_SIZE, stride = 1, dilation_rate = 1, act = None, name = 'conva_250_c') # 250
        features = tl.layers.BatchNormLayer(features, beta_init = w_init, gamma_init = w_init, is_train = is_train, name='bna3_c')
        features = PReluLayer(features, channel_shared = True, name='conv2a_relu_c')

        features = Conv1d(features, 32, KERNEL_SIZE, stride = 1, dilation_rate = 1, act = None, name = 'conv_250') # 250
        features = tl.layers.BatchNormLayer(features, beta_init = w_init, gamma_init = w_init, is_train = is_train, name='bn4')
        features = PReluLayer(features, channel_shared = True, name='conv2_relu')

        features = ElementwiseLayer([features, con_features], tf.add, name = 'elem_add')
        
        features = Conv1d(features, 32, KERNEL_SIZE, stride = 1, dilation_rate = 1, act = None, name = 'conv_125') # 125
        features = tl.layers.BatchNormLayer(features, beta_init = w_init, gamma_init = w_init, is_train = is_train, name='bn5')
        features = PReluLayer(features, channel_shared = True, name='conv3_relu')
        #sequences = Conv1d(sequences, 32, KERNEL_SIZE, stride = 4, dilation_rate = 1, act = act, name = 'conv_63') # 125
        
        #sequences = Conv1d(sequences, 32, KERNEL_SIZE, stride = 4, dilation_rate = 1, act = act, name = 'conv_31') # 125
        
        
        # stacking 3 bi-directiona,l lstm here
        
        features = BiRNNLayer(features, cell_fn = tf.contrib.rnn.LSTMCell, n_hidden = config.TRAIN.RNN_HIDDEN, n_steps = config.TRAIN.RNN_STEPS + 1, return_last = False, name = 'bi1')
        #features = PReluLayer(features, channel_shared = True, name='prelu1')
        #features = BiRNNLayer(features, cell_fn = tf.contrib.rnn.LSTMCell, n_hidden = config.TRAIN.RNN_HIDDEN, n_steps = config.TRAIN.RNN_STEPS + 1, return_last = False, name = 'bi2')
        #
        features = SelfAttentionLayer(features, 8 , 128,name='self-attention')
        '''

        features = Conv1d(sequences, 32, KERNEL_SIZE, stride = 2, dilation_rate = 1, act = None, name = 'conv1')
                
        features = tl.layers.BatchNormLayer(features, beta_init = w_init, gamma_init = w_init, is_train = is_train, name='bn1')
        
        features = PReluLayer(features, channel_shared = True, name='conv1_relu')

        features = Conv1d(features, 64, KERNEL_SIZE, stride = 2, act = None, name = 'conv1_stride')
        
        features = tl.layers.BatchNormLayer(features, beta_init = w_init, gamma_init = w_init, is_train = is_train, name='bn2')

        features = PReluLayer(features, channel_shared = True, name='conv2_relu')

        features = Conv1d(features, 64, KERNEL_SIZE, stride = 2, act = None, name = 'conv2_stride')
        
        features = tl.layers.BatchNormLayer(features, beta_init = w_init, gamma_init = w_init, is_train = is_train, name='bn3')

        features = PReluLayer(features, channel_shared = True, name='conv3_relu')
        '''
        
        return features, feature1.outputs



if __name__ == '__main__': 
    '''
    test model building
    '''
    print('model testing!')
    sequences = tf.placeholder(tf.float32, [None, config.TRAIN.TIME_STEPS, config.TRAIN.EMBED_DIM])
    selecting,_ = sharedFeatureExtractor(sequences,'extrator')
    category= classifier(selecting, 'classifier')
    #print(category.all_params)
    print('printing layers')
    print(category.all_params)
    #category.print_params(False)
