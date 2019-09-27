import sys
import os
from motif import *
from model import *
from easydict import EasyDict as edict
import tensorflow as tf
from util import *
import numpy as np
from config import *
from main import M6ANetShare
import os
import os
import tensorlayer as tl
from motif import *
from model import *
from keras.utils import to_categorical
import logging
import random
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter   ### 今天的主角

seed = int(sys.argv[1])

random.seed(seed)
np.random.seed(seed)
logger = logging.getLogger()
logger.setLevel(logging.ERROR)


dirs = '~/Research/m6a_prediction/motif/weblogo/mass/%s/filter%03d_logo/'
motifdir = '/home/xiongyuanpeng/motif_databases/'
sample_names =  ['panTro4','rheMac8','rn5','susScr3','danRer10','hg19','mm10',]
meme_dic = {'hg19': f' {motifdir}CISBP-RNA/Homo_sapiens.dna_encoded.meme {motifdir}/JASPAR/JASPAR2018_SPLICE.meme {motifdir}/JASPAR/JASPAR2018_CNE.meme {motifdir}/JASPAR/JASPAR2018_PHYLOFACTS.meme {motifdir}/JASPAR/JASPAR2018_POLII.meme {motifdir}/JASPAR/JASPAR2018_CORE_redundant.meme {motifdir}/HUMAN/HOCOMOCOv11_full_HUMAN_mono_meme_format.meme',\
'panTro4':f' {motifdir}/MOUSE/HOCOMOCOv11_full_MOUSE_mono_meme_format.meme {motifdir}/CISBP-RNA/Pan_troglodytes.dna_encoded.meme {motifdir}/JASPAR/JASPAR2018_PHYLOFACTS.meme {motifdir}/JASPAR/JASPAR2018_POLII.meme {motifdir}/JASPAR/JASPAR2018_CORE_redundant.meme {motifdir}/CISBP-RNA/Macaca_mulatta.dna_encoded.meme',\
'rheMac8':f' {motifdir}/JASPAR/JASPAR2018_CORE_redundant.meme {motifdir}/CISBP-RNA/Macaca_mulatta.dna_encoded.meme {motifdir}/JASPAR/JASPAR2018_PHYLOFACTS.meme {motifdir}/JASPAR/JASPAR2018_POLII.meme {motifdir}/CISBP-RNA/Macaca_mulatta.dna_encoded.meme',\
'mm10':f' {motifdir}/MOUSE/HOCOMOCOv11_full_MOUSE_mono_meme_format.meme {motifdir}/JASPAR/JASPAR2018_PHYLOFACTS.meme {motifdir}/JASPAR/JASPAR2018_CORE_redundant.meme {motifdir}/JASPAR/JASPAR2018_POLII.meme {motifdir}/CISBP-RNA/Mus_musculus.dna_encoded.meme',\
'rn5':f' {motifdir}/JASPAR/JASPAR2018_CORE_redundant.meme {motifdir}/CISBP-RNA/Rattus_norvegicus.dna_encoded.meme {motifdir}/JASPAR/JASPAR2018_POLII.meme {motifdir}/JASPAR/JASPAR2018_PHYLOFACTS.meme',\
'susScr3':f' {motifdir}/JASPAR/JASPAR2018_CORE_redundant.meme {motifdir}/CISBP-RNA/Sus_scrofa.dna_encoded.meme {motifdir}/JASPAR/JASPAR2018_POLII.meme {motifdir}/JASPAR/JASPAR2018_PHYLOFACTS.meme',\
'danRer10':f' {motifdir}/CISBP-RNA/Danio_rerio.dna_encoded.meme {motifdir}/JASPAR/JASPAR2018_CORE_redundant.meme {motifdir}/JASPAR/JASPAR2018_POLII.meme',\
}

def gen():
    index = 0
    while True:
        if index > len(lines):
            return
        start = index 
        end = index + batch_size
        
        index = end
        yield readList(lines[start:end])
        
'''
id = sample_names[int(sys.argv[1])]
#motif_dir = '/home/xiongyuanpeng/motif_databases/'
for i in range(300):
    dirname = dirs%(id,i) 
    motif_str = meme_dic[id]
    cmd = f'tomtom -oc {dirname}tomtom -dist pearson -thresh 0.05 {dirname}/meme.txt  {motif_str}' 
    print(cmd)
    os.system(cmd)
    #exit(o)
'''

gpu_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
gpu_config.gpu_options.allow_growth = True
sess = tf.Session(config=gpu_config)
tl.layers.initialize_global_variables(sess)

#print(filter_weights.shape)
#exit()
#dirs = '~/Research/m6a_prediction/motif/weblogo/mass/%s/filter%03d_logo/'

num_targets = len(sample_names)

global_filter_outs = []
seq_targets = []

for si in range(num_targets):
    sample = sample_names[si]
    print('processing ', sample)
    file_name = '/home/xiongyuanpeng/Research/m6a_prediction/data/sequence_samples/positive_samples/test/%s_pos_seqs_1000_100_self_processed'%(sample)
    lines = open(file_name).readlines()
    lines = random.sample(lines, 2000)
    batch_size = 64
    #print(len(lines))  
    max_iter = int(len(lines)/batch_size) + 1
    #inputs = readList(seqs)

    #x = f'/home/xiongyuanpeng/Research/m6a_prediction/data/processed_transcriptomes/{species}_{types}.seqs.pros'

    print('loading files...')
    #lines = [line.split()[-2] for line in open(x).readlines()]
    print('file loaded')
    #gens = gen(lines, batch_size)
    ds = tf.data.Dataset.from_generator(
        gen, tf.float32, tf.TensorShape([None, 101,4]))
    iters = ds.make_one_shot_iterator()
    t_sequences = iters.get_next()

    share_features, feature1, _ =  sharedFeatureExtractor2(t_sequences, 'extractor', is_train = False, reuse=tf.AUTO_REUSE)
    predict_score = classifier(share_features, 'test',reuse = tf.AUTO_REUSE, is_train = False)
    out = tf.nn.softmax(predict_score.outputs)
    feature1 = feature1.outputs
    classes_name = f'../checkpoint/m6aNet_classese_{sample}_rnn64_kerel18_18_18_fc128_lr0.01_classnum_fixed_drop0.5_mass_full_data_multi-task_7.npz'
    model_name = f'../checkpoint/m6aNet_feature_rnn64_kerel18_18_18_fc128_lr0.01_classnum_fixed_drop0.5_mass_full_data_multi-task_7.npz'
    #model_name = f'../checkpoint/m6aNet_feature_rnn64_kerel18_18_18_fc128_lr0.01_classnum_fixed_drop0.5_mass-single_full_data_{species}.npz'
    #classes_name = f'../checkpoint/m6aNet_classese_{species}_rnn64_kerel18_18_18_fc128_lr0.01_classnum_fixed_drop0.5_mass-single_full_data_{species}.npz'
    tl.files.load_and_assign_npz(sess = sess, network = share_features, name = model_name)
    tl.files.load_and_assign_npz(sess = sess, network = predict_score, name = classes_name)
    #if not tl.files.load_and_assign_npz(sess = sess, network = share_features, name = '../checkpoint/m6aNet_feature_rnn64_kerel18_18_18_fc128_lr0.01_classnum_fixed_drop0.5_mass_full_data_multi-task_7.npz'):
    #    print('no file loaded')
    #    exit()
        
    extractor = tl.layers.get_variables_with_name('extractor', True, True)
    filter_weights = extractor[0].eval(session=sess)
    
    filter_outs = []
    #test = np.random.randint(0,2,size= 32)
    print('running output')
    for idx in range(max_iter+1):
        #input_np = inputs_np[i*32:min((i+1)*32,inputs_np.shape[0])]
        #feed_dict[t_sequences] = input_np
        try:
            res,scores = sess.run([feature1, out])
        except:
            continue
        if True:
            if idx >= max_iter:
                continue
                #filter_outs.extend(res)
            #res_filt = []
            for i in range(len(scores)):
                #print(scores[i])
                if scores[i][1] > 0.:
                    #res_filt.append(res)
                    global_filter_outs.append(res[i])
                    seq_targets.append(si)
            
            percent = (idx + 1) * 50 / max_iter
            num_arrow = int(percent)
            num_line = 50 - num_arrow
            progress_bar = '[' + '>' * num_arrow + '-' * num_line + ']' + '%.2f%% \r' % (percent*2) 
            sys.stdout.write(progress_bar)
            sys.stdout.flush()
                
    print('\n')
    #global_filter_outs.extend(filter_outs)
    #seq_targets = seq_targets + [si,]*len(filter_outs)
    filter_outs  = (np.array(filter_outs))
    #print(filter_outs.shape)
    meme_db_files = meme_dic[sample].split()  # list of meme database
    
    motif_protein = get_motif_proteins(meme_db_files)
    num_filters = 300
    filter_names = ['f%d'%fi for fi in range(num_filters)]
    #filter_names = name_filters(num_filters, '%s/tomtom/tomtom.tsv'%options.out_dir, options.meme_db)
    filter_motifs = {}
    for i in range(num_filters):
        motif_name =  '/home/xiongyuanpeng/Research/m6a_prediction/motif/weblogo/mass/%s/filter%03d_logo/tomtom/tomtom.tsv'%(sample, i)
        try:
            for line in open(motif_name):
                lines = line.split()
                if(len(lines)== 0):
                    continue
                if (lines[0][0] is not 'Q') and (lines[0][0] is not '#'):
                    #print(lines)
                    q_val = float(lines[5])
                    motif_id = lines[1]
                    filter_motifs.setdefault(i,[]).append((q_val, motif_id))
        except:
            filter_motifs.setdefault(i,[])
    for fi in filter_motifs:
        try:
            top_motif = sorted(filter_motifs[fi])[0][1]
            q_val = sorted(filter_motifs[fi])[0][0]
            filter_names[fi] += '_%s_%.4f' % (motif_protein[top_motif], q_val)
        except:
            continue
    filter_names = np.array(filter_names)
    #print(filter_names)
    #exit()
    #################################################################
    # print a table of information
    #################################################################
    
    basset_dir = '/home/xiongyuanpeng/Research/m6a_prediction/motif/weblogo/mass/%s/basset_analysis/'%sample
    tl.files.exists_or_mkdir(basset_dir)
    
    table_out = open('%s/table.txt'%basset_dir, 'w')

    # print header for later panda reading
    header_cols = ('', 'consensus', 'annotation', 'ic', 'mean', 'std','q-value')
    table_out.write('%3s  %19s  %10s  %5s  %6s  %6s  %8s\n' % header_cols)
    
    print('generating activation dense map')
    for f in range(num_filters):
        # collapse to a consensus motif
        consensus = filter_motif(filter_weights[:,:,f])

        # grab annotation
        annotation = '.'
        q_val = '1e14'
        name_pieces = filter_names[f].split('_')
        if len(name_pieces) > 1:
            annotation = name_pieces[-2]
            q_val = name_pieces[-1]

        # plot density of filter output scores
        #ax = plt.gca()  # 获取当前图像的坐标轴信息
        #ax.xaxis.get_major_formatter().set_powerlimits((0,1)) # 将坐标轴的base number设置为一位。
        #ax.yaxis.get_major_formatter().set_powerlimits((0,1)) # 将坐标轴的base number设置为一位。
        
        #fmean, fstd = plot_score_density(np.ravel(filter_outs[:,:,f]**2), '%s/filter%d_dens.pdf' % (basset_dir, f))
        fmean = fstd = 0
        row_cols = (f, consensus, annotation, 0, fmean, fstd, q_val)
        table_out.write('%-3d  %19s  %10s  %5.2f  %6.4f  %6.4f  %8s\n' % row_cols)
        
        percent = (f + 1) * 50 / num_filters
        num_arrow = int(percent)
        num_line = 50 - num_arrow
        progress_bar = '[' + '>' * num_arrow + '-' * num_line + ']' + '%.2f%% \r' % (percent*2) 
        sys.stdout.write(progress_bar)
        sys.stdout.flush()
    
    print('\n')
    table_out.close()
    
    

    
#################################################################
# global filter plots
################################################################
# 传入前要做一次换轴

new_dir = '/home/xiongyuanpeng/Research/m6a_prediction/motif/weblogo/mass/'
#print(np.array(global_filter_outs).shape)
tl.files.exists_or_mkdir(new_dir)
global_filter_outs = np.array(global_filter_outs)
print(global_filter_outs.shape) 
filter_outs = np.transpose(global_filter_outs, (0, 2, 1))

target_names = sample_names
seq_targets = to_categorical(seq_targets)
if True:
    if True:
        # plot filter-sequence heatmap
        #plot_filter_seq_heat(filter_outs, '%s/filter_seqs.pdf'%new_dir)

        # plot filter-segment heatmap
        #plot_filter_seg_heat(filter_outs, '%s/filter_segs.pdf'%new_dir)
        #plot_filter_seg_heat(filter_outs, '%s/filter_segs_raw.pdf'%new_dir, whiten=False)

        # plot filter-target correlation heatmap
        plot_target_corr(filter_outs, seq_targets, filter_names, target_names, '%s/filter_target_cors_mean_%d.pdf'%(new_dir, seed), 'mean')
        plot_target_corr(filter_outs, seq_targets, filter_names, target_names, '%s/filter_target_cors_max_%d.pdf'%(new_dir, seed), 'max')
'''
'''
