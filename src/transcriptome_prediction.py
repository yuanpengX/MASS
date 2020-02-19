import tensorflow as tf
from util import *
# generate inputs
from util import *
import numpy as np
from config import *
from main import M6ANetShare
import os
import tensorlayer as tl
from motif import *
from model import *
from matplotlib import pyplot as plt
import seaborn as sns
import sys




def gen():
    index = 0
    while True:
        if index >= len(lines):
            return 
        start = index 
        end = index + batch_size
        
        index += batch_size
        yield readList(lines[start:end])
samples = ['hg19','mm10','rheMac8','panTro4','susScr3','rn5','danRer10']
t = ['mRNA','transcripts','genome']



os.environ['CUDA_VISIBLE_DEVICES']= sys.argv[1]
species = samples[int(sys.argv[2])]
types = t[int(sys.argv[3])]#'mRNA'
model = sys.argv[4]#'mass-single'#'deepm6aseq'

print(species,' ', types)
if types == 'genome':
    x = f'/data/group/m6A/genome_seqs/{species}.genome.pros' 
else:
    x = f'/data/group/m6A/processed_transcriptomes/{species}_{types}.seqs.pros'
batch_size = 32
print('loading files...')
lines = [line.split()[-2] for line in open(x).readlines()]
print('file loaded')
#gens = gen(lines, batch_size)
ds = tf.data.Dataset.from_generator(
    gen, tf.float32, tf.TensorShape([None, 101,4]))
iters = ds.make_one_shot_iterator()
t_sequences = iters.get_next()


if model == 'deepm6aseq':
    share_features, _ = DeepM6ASeq_pre(t_sequences, 'extractor', reuse = False, is_train=False)
    predict_score = DeepM6ASeq(share_features, 'tmo',reuse = False, is_train = False)
else:
    share_features, _, _ = sharedFeatureExtractor2(t_sequences , 'extractor', is_train = False)

    predict_score = classifier(share_features, 'tmp', is_train = False)
out = tf.nn.softmax(predict_score.outputs)
gpu_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
gpu_config.gpu_options.allow_growth = True
        
sess = tf.Session(config=gpu_config)
        
#sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
tl.layers.initialize_global_variables(sess)

if model == 'deepm6aseq':        
    model_name = f'/data/group/m6A/checkpoint/m6aNet_feature_rnn64_kerel18_18_18_fc128_lr0.01_classnum_fixed_drop0.5_deepm6aseq_full_data_{species}.npz'
    classes_name = f'/data/group/m6A/checkpoint/m6aNet_classese_{species}_rnn64_kerel18_18_18_fc128_lr0.01_classnum_fixed_drop0.5_deepm6aseq_full_data_{species}.npz'
else:
    if model == 'mass':
        model_name = '/data/group/m6A/checkpoint/m6aNet_feature_rnn64_kerel18_18_18_fc128_lr0.01_classnum_fixed_drop0.5_mass_full_data_multi-task_7.npz'
        classes_name = '/data/group/m6A/checkpoint/m6aNet_classese_%s_rnn64_kerel18_18_18_fc128_lr0.01_classnum_fixed_drop0.5_mass_full_data_multi-task_7.npz'% species
    else:
        model_name = f'/data/group/m6A/checkpoint/m6aNet_feature_rnn64_kerel18_18_18_fc128_lr0.01_classnum_fixed_drop0.5_mass-single_full_data_{species}.npz'
        classes_name = f'/data/group/m6A/checkpoint/m6aNet_classese_{species}_rnn64_kerel18_18_18_fc128_lr0.01_classnum_fixed_drop0.5_mass-single_full_data_{species}.npz'

tl.files.load_and_assign_npz(sess = sess, network = share_features, name = model_name)
tl.files.load_and_assign_npz(sess = sess, network = predict_score, name = classes_name)

max_iter = int(len(lines)/batch_size)+1
scores = []
import tensorlayer as tl
if types == 'genome':

    tl.files.exists_or_mkdir(f'/data/group/m6A/genome_seqs/{model}/')

    fp = open(f'/data/group/m6A/genome_seqs/{model}/{species}.{types}.res','w')

else: 
    tl.files.exists_or_mkdir(f'/data/group/m6A/processed_transcriptomes/{model}/')
    fp = open(f'/data/group/m6A/processed_transcriptomes/{model}/{species}_{types}.res','w')

for idx in range(max_iter+1):
    try:
        scores = sess.run(out)
        for s in scores:
            fp.write('%f\n'%s[1])

        percent = (idx + 1) * 50 / max_iter
        num_arrow = int(percent)
        num_line = 50 - num_arrow
        progress_bar = '[' + '>' * num_arrow + '-' * num_line + ']' + '%.2f%% \r' % (percent*2) 
        sys.stdout.write(progress_bar)
        sys.stdout.flush()
    except:
        pass
#np.savetxt('score.txt',np.array(scores))
