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

sample_names =  ['hg19','panTro4','rheMac8','mm10','rn5','susScr3','danRer10',]

os.environ['CUDA_VISIBLE_DEVICES']= sys.argv[2]
sample = sample_names[int(sys.argv[1])]
#tl.global_flag['name'] =  'm6aNet_feature_rnn64_kerel18_18_18_fc128_lr0.01_classnum_fixed_drop0.5_mass_full_data_{}'.format(sample)
#tl.files.exists_or_mkdir('../motif/head/%s/'%tl.global_flag['name'])
#tl.files.exists_or_mkdir('../motif/possum/%s/'%tl.global_flag['name'])
#m6anet = M6ANetShare(None)
t_sequences = tf.placeholder(tf.float32, [None, config.TRAIN.TIME_STEPS, config.TRAIN.EMBED_DIM])
model = 'mass'#'deepm6aseq'
#feature,_ = DeepM6ASeq_pre(t_sequences, 'deepm6aseq', is_train = False)
#feature, feature1 = DeepM6ASeq(feature,'extractor', is_train = False)
#feature1 = feature1.outputs
feature, feature1,_ =  sharedFeatureExtractor2(t_sequences, 'extractor', is_train = False)
predict_score, feature1 = classifier(feature, 'fa',is_train = False)

feature1 = tf.nn.relu(feature1.outputs)

gpu_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
gpu_config.gpu_options.allow_growth = True
sess = tf.Session(config=gpu_config)
tl.layers.initialize_global_variables(sess)
#if not tl.files.load_and_assign_npz(sess = sess, network = feature, name = f'../checkpoint/m6aNet_feature_rnn64_kerel18_18_18_fc128_lr0.01_classnum_fixed_drop0.5_{model}_full_data_%s.npz'%(sample)):
#    exit()

if not tl.files.load_and_assign_npz(sess = sess, network = feature, name = '../checkpoint/m6aNet_feature_rnn64_kerel18_18_18_fc128_lr0.01_classnum_fixed_drop0.5_mass_full_data_multi-task_7.npz'.format(sample)):
    exit()

classes_name = f'../checkpoint/m6aNet_classese_{sample}_rnn64_kerel18_18_18_fc128_lr0.01_classnum_fixed_drop0.5_mass_full_data_multi-task_7.npz'

if not tl.files.load_and_assign_npz(sess = sess, network = predict_score, name = classes_name):
    exit()

extractor = tl.layers.get_variables_with_name('extractor', True, True)
kernel = extractor[0].eval(session=sess)
num_filter = 32

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
    tl.files.exists_or_mkdir(f'../result/{model}-spe')
    tl.files.exists_or_mkdir(f'../result//{model}-spe/{sample}/')
    file_name = '../data/sequence_samples/positive_samples/test/%s_pos_seqs_1000_100_self_processed'%(sample)
    seqs = open(file_name).readlines()
    print(len(seqs))
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
    print('generating weblogo:')
    for i in range(num_filter):
        plot_filter_logo(filter_out[:,:,i], filter_size, seqs, f'../result/{model}-spe/{sample}/filter%03d_logo'%(i),maxpct_t=0.5)
