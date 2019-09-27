# generate inputs
from util import *
import numpy as np
from config import *
#from main import M6ANetShare
import os
import os
import tensorlayer as tl
from motif import *
from model import *
import sys
import tensorflow
import random 
import tensorflow as tf
import pickle
np.random.seed(2)
random.seed(2)
tf.set_random_seed(2)
class Transpose(Layer):
    """A layer that transposes the dimension of a tensor.

    See `tf.transpose() <https://www.tensorflow.org/api_docs/python/tf/transpose>`__ .

    Parameters
    ----------
    perm: list of int
        The permutation of the dimensions, similar with ``numpy.transpose``.
        If None, it is set to (n-1...0), where n is the rank of the input tensor.
    conjugate: bool
        By default False. If True, returns the complex conjugate of complex numbers (and transposed)
        For example [[1+1j, 2+2j]] --> [[1-1j], [2-2j]]
    name : str
        A unique layer name.

    Examples
    ----------
    >>> x = tl.layers.Input([8, 4, 3], name='input')
    >>> y = tl.layers.Transpose(perm=[0, 2, 1], conjugate=False, name='trans')(x)
    (8, 3, 4)

    """

    def __init__(self, perm=None, conjugate=False, name=None):  #'transpose'):
        super(Transpose, self).__init__(name)
        self.perm = perm
        self.conjugate = conjugate

        logging.info("Transpose  %s: perm: %s, conjugate: %s" % (self.name, self.perm, self.conjugate))

        self.build()
        self._built = True

    def __repr__(self):
        s = '{classname}('
        s += 'perm={perm},'
        s += 'conjugate={conjugate},'
        s += 'name=\'{name}\''
        s += ')'
        return s.format(classname=self.__class__.__name__, **self.__dict__)

    def build(self, inputs_shape=None):
        pass

    # @tf.function
    def forward(self, inputs):
        outputs = tf.transpose(a=inputs, perm=self.perm, conjugate=self.conjugate, name=self.name)
        return outputs

sample_names =  ['hg19','panTro4','rheMac8','mm10','rn5','susScr3','danRer10',]

os.environ['CUDA_VISIBLE_DEVICES']= sys.argv[2]
sample = sample_names[int(sys.argv[1])]
pos = int(sys.argv[3])

gpu_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
gpu_config.gpu_options.allow_growth = True
sess = tf.Session(config=gpu_config)

#tl.global_flag['name'] =  'm6aNet_feature_rnn64_kerel18_18_18_fc128_lr0.01_classnum_fixed_drop0.5_mass_full_data_{}'.format(sample)
#tl.files.exists_or_mkdir('../motif/head/%s/'%tl.global_flag['name'])
#tl.files.exists_or_mkdir('../motif/possum/%s/'%tl.global_flag['name'])
#m6anet = M6ANetShare(None)
t_sequences = tf.placeholder(tf.float32, [None, config.TRAIN.TIME_STEPS, config.TRAIN.EMBED_DIM])
models = ['mass', 'mass-single','deepm6aseq']

model = models[int(sys.argv[4])]#'mass-single'
#fins = ['fin', 'shapre']

fin = 'fin'

if not (model == 'mass'):
    if model == 'mass-single':
        features, feature1, final_features =  sharedFeatureExtractor2(t_sequences, 'extractor', is_train = False)
        cate, feature = classifier(features, sample, is_train = False)
    else:
        features,final_features = DeepM6ASeq_pre(t_sequences, 'deepmaseq', is_train = False)
        cate, feature = DeepM6ASeq(features,'extractor', is_train = False)
    if fin == 'fin':

        feature = feature
    else:
        feature = final_features

    tl.layers.initialize_global_variables(sess)
    #feature1 = feature1.outputs
    if not tl.files.load_and_assign_npz(sess = sess, network = features, name = f'../checkpoint/m6aNet_feature_rnn64_kerel18_18_18_fc128_lr0.01_classnum_fixed_drop0.5_{model}_full_data_%s.npz'%(sample)):
        exit()

    if not tl.files.load_and_assign_npz(sess = sess, network = cate, name = f'../checkpoint/m6aNet_classese_{sample}_rnn64_kerel18_18_18_fc128_lr0.01_classnum_fixed_drop0.5_{model}_full_data_%s.npz'%(sample)):
        exit()
    
else:
    features, feature1, final_features =  sharedFeatureExtractor2(t_sequences, 'extractor', is_train = False)

    cate, feature = classifier(features, sample, is_train = False)
    if fin == 'fin':
        feature = feature
    else:
        feature = final_features

    tl.layers.initialize_global_variables(sess)

    if not tl.files.load_and_assign_npz(sess = sess, network = features, name = '../checkpoint/m6aNet_feature_rnn64_kerel18_18_18_fc128_lr0.01_classnum_fixed_drop0.5_mass_full_data_multi-task_7.npz'):
        exit()
    if not tl.files.load_and_assign_npz(sess = sess, network = cate, name = f'../checkpoint/m6aNet_classese_{sample}_rnn64_kerel18_18_18_fc128_lr0.01_classnum_fixed_drop0.5_mass_full_data_multi-task_7.npz'):
        exit()
#feature1 = tf.nn.relu(feature1)

def res(inputs):
    return tf.transpose(a=inputs, perm = [0, 2, 1], conjugate=False, name='res')

feature = tl.layers.LambdaLayer(feature, fn = res, name='reshape')
#LambdaLayer(feature)#Transpose(perm=[0, 2, 1], conjugate=False, name='trans')(feature)
feature1 = FlattenLayer(feature, name='flatten_features').outputs

extractor = tl.layers.get_variables_with_name('extractor', True, True)
kernel = extractor[0].eval(session=sess)
num_filter = 300

filter_size = 18

'''
for i in range(num_filter):
    plot_filter_heat(kernel[:,:,i].T, '../motif/head/%s/filter%d_head.pdf'%(tl.global_flag['name'],i))
    filter_possum(kernel[:,:,i].T, 'filter%d'%i, '../motif/possum/%s/filter%d_possum.txt'%(tl.global_flag['name'],i), False)
'''

'''
for i in range(config.TRAIN.CLASSES - 1):
    sample = config.TRAIN.sample_names[i]
    feed_dict[m6anet.targets[sample]] = np.random.randint(0,2,32)
'''        
feed_dict = {}
if True:
    #sample = 'rheMac8'#,'#config.sample_names[id]
    print('Processing:', sample)
    tl.files.exists_or_mkdir(f'/data/group/m6A//{model}')
    tl.files.exists_or_mkdir(f'/data/group/m6A/{model}/{sample}/')
    if pos == 1:
        file_name = '/home/xiongyuanpeng/Research/m6a_prediction/data/sequence_samples/positive_samples/test/%s_pos_seqs_1000_100_self_processed'%(sample)
    else:
        file_name = '/home/xiongyuanpeng/Research/m6a_prediction/data/sequence_samples/negative_samples/test/%s_neg_seqs_1000_100_self_processed'%(sample)
    seqs = open(file_name).readlines()
    seqs  =  random.sample(seqs, 2000)

    #print(len(seqs))
    inputs = readList(seqs)
    inputs_np = np.array(inputs)
    print(inputs_np.shape)
    filter_out = []
    test = np.random.randint(0,2,size= 32)
    print('running output')
    for i in range(int(inputs_np.shape[0]/32)+1):
        input_np = inputs_np[i*32:min((i+1)*32,inputs_np.shape[0])]
        feed_dict[t_sequences] = input_np
        filter_out.extend(sess.run(feature1, feed_dict = feed_dict))
    filter_out = np.array(filter_out)
    sess.close()
    print(filter_out.shape)
    pickle.dump(filter_out, open(f'result/{model}_{fin}_{sample}_{pos}.bin','wb'))
    #
    #print('generating weblogo:')
    #for i in range(num_filter):
    #    plot_filter_logo(filter_out[:,:,i], filter_size, seqs, f'/data/group/m6A/{model}/{sample}/filter%03d_logo'%(i),maxpct_t=0.5)
