import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import tensorlayer as tl
from sklearn import preprocessing
from scipy.stats import spearmanr 
import pandas as pd

'''
These code comes from Basset:Deep convolutional neural networks for DNA sequence analysis
https://github.com/davek44/Basset/blob/master/src/basset_motifs.py
I made some modification to fit my own dataset.
'''


def get_motif_proteins(meme_db_files):
    ''' Hash motif_id's to protein names using the MEME DB file '''
    motif_protein = {}
    for meme_db_file in meme_db_files:
        for line in open(meme_db_file):
            a = line.split()
            if len(a) > 0 and a[0] == 'MOTIF':
                if len(a) == 2:
                    motif_protein[a[1]] = a[1]
                    continue
                if a[2][0] == '(':
                    motif_protein[a[1]] = a[2][1:a[2].find(')')]
                else:
                    motif_protein[a[1]] = a[2]
    return motif_protein
def info_content(pwm, transpose=False, bg_gc=0.415):
    ''' Compute PWM information content.
    In the original analysis, I used a bg_gc=0.5. For any
    future analysis, I ought to switch to the true hg19
    value of 0.415.
    '''
    pseudoc = 1e-9

    if transpose:
        pwm = np.transpose(pwm)

    bg_pwm = [1-bg_gc, bg_gc, bg_gc, 1-bg_gc]

    ic = 0
    for i in range(pwm.shape[0]):
        for j in range(4):
            # ic += 0.5 + pwm[i][j]*np.log2(pseudoc+pwm[i][j])
            ic += -bg_pwm[j]*np.log2(bg_pwm[j]) + pwm[i][j]*np.log2(pseudoc+pwm[i][j])

    return ic
def make_filter_pwm(filter_fasta):
    ''' Make a PWM for this filter from its top hits '''

    nts = {'A':0, 'T':1, 'C':2, 'G':3}
    pwm_counts = []
    nsites = 4 # pseudocounts
    for line in open(filter_fasta):
        if line[0] != '>':
            seq = line.rstrip()
            nsites += 1
            if len(pwm_counts) == 0:
                # initialize with the length
                for i in range(len(seq)):
                    pwm_counts.append(np.array([1.0]*4))

            # count
            for i in range(len(seq)):
                try:
                    pwm_counts[i][nts[seq[i]]] += 1
                except KeyError:
                    pwm_counts[i] += np.array([0.25]*4)

    # normalize
    pwm_freqs = []
    for i in range(len(pwm_counts)):
        pwm_freqs.append([pwm_counts[i][j]/float(nsites) for j in range(4)])

    return np.array(pwm_freqs), nsites-4
def plot_filter_heat(param_matrix, out_pdf):
    param_range = abs(param_matrix).max()

    sns.set(font_scale=2)
    plt.figure(figsize=(param_matrix.shape[1], 4))
    sns.heatmap(param_matrix, cmap='PRGn', linewidths=0.2, vmin=-param_range, vmax=param_range)
    ax = plt.gca()
    ax.set_xticklabels(range(1,param_matrix.shape[1]+1))
    ax.set_yticklabels('ATCG', rotation='horizontal') # , size=10)
    plt.savefig(out_pdf)
    plt.close()
    
def filter_possum(param_matrix, motif_id, possum_file, trim_filters=False, mult=200):
    # possible trim
    trim_start = 0
    trim_end = param_matrix.shape[1]-1
    trim_t = 0.3
    if trim_filters:
        # trim PWM of uninformative prefix
        while trim_start < param_matrix.shape[1] and np.max(param_matrix[:,trim_start]) - np.min(param_matrix[:,trim_start]) < trim_t:
            trim_start += 1

        # trim PWM of uninformative suffix
        while trim_end >= 0 and np.max(param_matrix[:,trim_end]) - np.min(param_matrix[:,trim_end]) < trim_t:
            trim_end -= 1

    if trim_start < trim_end:
        fp = open(possum_file, 'w')
        fp.write('BEGIN GROUP\n')#, possum_out)# >> possum_out, 'BEGIN GROUP'
        fp.write('BEGIN FLOAT\n')#,possum_out) #>> possum_out, 
        fp.write('ID %s\n'%motif_id)# % motif_id,possum_out)# >> possum_out, 'ID %s' % motif_id
        fp.write('AP DNA\n')#,possum_out)# >> possum_out, 'AP DNA'
        fp.write('LE %d\n' % (trim_end+1-trim_start))#,possum_out)# >> possum_out, 'LE %d' % (trim_end+1-trim_start)
        for ci in range(trim_start,trim_end+1):
            fp.write('MA %s\n' % ' '.join(['%.2f'%(mult*n) for n in param_matrix[:,ci]]))#,possum_out)# >> possum_out, 'MA %s' % ' '.join(['%.2f'%(mult*n) for n in param_matrix[:,ci]])
        fp.write('END\n')#,possum_out) #>> possum_out, 'END'
        fp.write('END\n')#,possum_out)#print >> possum_out, 'END'

        fp.close()
import subprocess
weblogo_opts = '-X NO -Y NO --errorbars NO --fineprint "" --resolution 600'
weblogo_opts += ' -C "#CB2026" A A'
weblogo_opts += ' -C "#34459C" C C'
weblogo_opts += ' -C "#FBB116" G G'
weblogo_opts += ' -C "#0C8040" T T'

def plot_filter_logo(filter_outs, filter_size, seqs, out_prefix, raw_t=0, maxpct_t=None):
    #tl.files.exists_or_mkdir(out_prefix)
    #name = out_prefix + out_prefix.split('/')[-1]
    if maxpct_t:
        all_outs = np.ravel(filter_outs)
        all_outs_mean = all_outs.mean()
        all_outs_norm = all_outs - all_outs_mean
        raw_t = maxpct_t * all_outs_norm.max() + all_outs_mean

    # SAME padding
    pad_side = (filter_size - 1) // 2

    # print fasta file of positive outputs
    fp = open('%s.fa' % out_prefix, 'w')
    filter_count = 0
    for i in range(filter_outs.shape[0]):
        for j in range(pad_side, filter_outs.shape[1]-pad_side):
            if filter_outs[i,j] > raw_t:
                js = (j - pad_side)
                kmer = seqs[i][js:js+filter_size]
                if len(kmer.strip()) < filter_size:
                    continue
                fp.write('>%d_%d\n' % (i,j))
                fp.write(kmer+'\n')
                filter_count += 1
    fp.close()
    '''
    # make weblogo
    if filter_count > 0:
        meme_cmd = f'meme {out_prefix}.fa -dna -oc {out_prefix} -nostatus -time 18000 -mod zoops -nmotifs 2 -minw 6 -maxw 50 -objfun classic -revcomp -markov_order 0'
        #meme_cmd = 'meme %s.fa -dna -mod zoops -pal -o %s -nmotifs 2'%(out_prefix, out_prefix)
        #weblogo_cmd = 'weblogo %s < %s.fa > %s.eps&' % (weblogo_opts, out_prefix, out_prefix)
        #print(weblogo_cmd)
        subprocess.call(meme_cmd, shell=True)
    '''
        
        
def meme_add(meme_out, f, filter_pwm, nsites, trim_filters=False):
    ''' Print a filter to the growing MEME file
    Attrs:
        meme_out : open file
        f (int) : filter index #
        filter_pwm (array) : filter PWM array
        nsites (int) : number of filter sites
    '''
    if not trim_filters:
        ic_start = 0
        ic_end = filter_pwm.shape[0]-1
    else:
        ic_t = 0.2

        # trim PWM of uninformative prefix
        ic_start = 0
        while ic_start < filter_pwm.shape[0] and info_content(filter_pwm[ic_start:ic_start+1]) < ic_t:
            ic_start += 1

        # trim PWM of uninformative suffix
        ic_end = filter_pwm.shape[0]-1
        while ic_end >= 0 and info_content(filter_pwm[ic_end:ic_end+1]) < ic_t:
            ic_end -= 1

    if ic_start < ic_end:
        meme_out.write('MOTIF filter%d\n' % f)# z3print >> 
        meme_out.write( 'letter-probability matrix: alength= 4 w= %d nsites= %d\n' % (ic_end-ic_start+1, nsites))#print >> 

        for i in range(ic_start, ic_end+1):
            meme_out.write( '%.4f %.4f %.4f %.4f\n' % tuple(filter_pwm[i]))#print >> 
        meme_out.write('\n')#print >> 


def meme_intro(meme_file, seqs):
    ''' Open MEME motif format file and print intro
    Attrs:
        meme_file (str) : filename
        seqs [str] : list of strings for obtaining background freqs
    Returns:
        mem_out : open MEME file
    '''
    nts = {'A':0, 'T':1, 'C':2, 'G':3}

    # count
    nt_counts = [1]*4
    for i in range(len(seqs)):
        for nt in seqs[i]:
            try:
                nt_counts[nts[nt]] += 1
            except KeyError:
                pass

    # normalize
    nt_sum = float(sum(nt_counts))
    nt_freqs = [nt_counts[i]/nt_sum for i in range(4)]

    # open file for writing
    meme_out = open(meme_file, 'w')

    # print intro material
    meme_out.write( 'MEME version 4\n')#print >> 
    meme_out.write('\n')#print >> 
    meme_out.write( 'ALPHABET= ACGT\n')#print >> 
    meme_out.write( '\n')#print >> 
    meme_out.write('Background letter frequencies:\n')#print >> 
    meme_out.write('A %.4f C %.4f G %.4f T %.4f\n' % tuple(nt_freqs))#print >> 
    meme_out.write('\n')#print >> 

    return meme_out


def name_filters(num_filters, tomtom_file, meme_db_file):
    ''' Name the filters using Tomtom matches.
    Attrs:
        num_filters (int) : total number of filters
        tomtom_file (str) : filename of Tomtom output table.
        meme_db_file (str) : filename of MEME db
    Returns:
        filter_names [str] :
    '''
    # name by number
    #filter_names = ['f%d'%fi for fi in range(num_filters)]

    # name by protein
    if tomtom_file is not None and meme_db_file is not None:
        motif_protein = get_motif_proteins(meme_db_file)

        # hash motifs and q-value's by filter
        filter_motifs = []
        try:
            tt_in = open(tomtom_file)
        except:
            return ''
        for line in tt_in:
            
            a = line.split()
            if  len(a)==0 or a[0].startswith('Q') or a[0].startswith('#'):
                continue
            #fi = int(a[0][6:])
            print(tomtom_file) 
            print(a)
            motif_id = a[1]
            pval = float(a[3])
            evals = float(a[4])
            qval = float(a[5])
            
            filter_motifs.append((evals, pval, qval,motif_id))

        tt_in.close()

        # assign filter's best match
        try:
            tmp = sorted(filter_motifs)[0]
            top_motif = tmp[-1]
            pval = tmp[1]
            evals = tmp[0]
            qval = tmp[2]
            filter_names = 'test_%s_%f_%f_%f' % (motif_protein[top_motif], pval, evals, qval)
        except:
            filter_names = ''#'test_test_test_test_test'
    return filter_names


################################################################################
# plot_target_corr
#
# Plot a clustered heatmap of correlations between filter activations and
# targets.
#
# Input
#  filter_outs:
#  filter_names:
#  target_names:
#  out_pdf:
################################################################################
def plot_target_corr(filter_outs, seq_targets, filter_names, target_names, out_pdf, seq_op='mean'):
    num_seqs = filter_outs.shape[0]
    num_targets = len(target_names)

    if seq_op == 'mean':
        filter_outs_seq = filter_outs.mean(axis=2)
    else:
        filter_outs_seq = filter_outs.max(axis=2)

    # std is sequence by filter.
    filter_seqs_std = filter_outs_seq.std(axis=0)
    filter_outs_seq = filter_outs_seq[:,filter_seqs_std > 0]
    filter_names_live = filter_names[filter_seqs_std > 0]

    filter_target_cors = np.zeros((len(filter_names_live),num_targets))
    for fi in range(len(filter_names_live)):
        for ti in range(num_targets):
            cor, p = spearmanr(filter_outs_seq[:,fi], seq_targets[:num_seqs,ti])
            filter_target_cors[fi,ti] = cor
    
    cor_df = pd.DataFrame(filter_target_cors, index=filter_names_live, columns=target_names)
    #print(cor_df)
    sns.set(font_scale=0.3)
    plt.figure()
    sns.clustermap(cor_df, cmap='BrBG', center=0, figsize=(8,10))
    plt.savefig(out_pdf)
    plt.close()


################################################################################
# plot_filter_seq_heat
#
# Plot a clustered heatmap of filter activations in
#
# Input
#  param_matrix: np.array of the filter's parameter matrix
#  out_pdf:
################################################################################
def plot_filter_seq_heat(filter_outs, out_pdf, whiten=True, drop_dead=True):
    # compute filter output means per sequence
    filter_seqs = filter_outs.mean(axis=2)

    # whiten
    if whiten:
        filter_seqs = preprocessing.scale(filter_seqs)

    # transpose
    filter_seqs = np.transpose(filter_seqs)

    if drop_dead:
        filter_stds = filter_seqs.std(axis=1)
        filter_seqs = filter_seqs[filter_stds > 0]

    # downsample sequences
    seqs_i = np.random.randint(0, filter_seqs.shape[1], 500)

    hmin = np.percentile(filter_seqs[:,seqs_i], 0.1)
    hmax = np.percentile(filter_seqs[:,seqs_i], 99.9)

    sns.set(font_scale=0.3)

    plt.figure()
    sns.clustermap(filter_seqs[:,seqs_i], row_cluster=True, col_cluster=True, linewidths=0, xticklabels=False, vmin=hmin, vmax=hmax)
    plt.savefig(out_pdf)
    #out_png = out_pdf[:-2] + 'ng'
    #plt.savefig(out_png, dpi=300)
    plt.close()


################################################################################
# plot_filter_seq_heat
#
# Plot a clustered heatmap of filter activations in sequence segments.
#
# Mean doesn't work well for the smaller segments for some reason, but taking
# the max looks OK. Still, similar motifs don't cluster quite as well as you
# might expect.
#
# Input
#  filter_outs
################################################################################
def plot_filter_seg_heat(filter_outs, out_pdf, whiten=True, drop_dead=True):
    b = filter_outs.shape[0]
    f = filter_outs.shape[1]
    l = filter_outs.shape[2]

    s = 5
    while l/float(s) - (l/s) > 0:
        s += 1
    print('%d segments of length %d' % (s,l/s))

    # split into multiple segments
    filter_outs_seg = np.reshape(filter_outs, (b, f, s, l/s))

    # mean across the segments
    filter_outs_mean = filter_outs_seg.max(axis=3)

    # break each segment into a new instance
    filter_seqs = np.reshape(np.swapaxes(filter_outs_mean, 2, 1), (s*b, f))

    # whiten
    if whiten:
        filter_seqs = preprocessing.scale(filter_seqs)

    # transpose
    filter_seqs = np.transpose(filter_seqs)

    if drop_dead:
        filter_stds = filter_seqs.std(axis=1)
        filter_seqs = filter_seqs[filter_stds > 0]

    # downsample sequences
    seqs_i = np.random.randint(0, filter_seqs.shape[1], 500)

    hmin = np.percentile(filter_seqs[:,seqs_i], 0.1)
    hmax = np.percentile(filter_seqs[:,seqs_i], 99.9)

    sns.set(font_scale=0.3)
    if whiten:
        dist = 'euclidean'
    else:
        dist = 'cosine'

    plt.figure()
    sns.clustermap(filter_seqs[:,seqs_i], metric=dist, row_cluster=True, col_cluster=True, linewidths=0, xticklabels=False, vmin=hmin, vmax=hmax)
    plt.savefig(out_pdf)
    #out_png = out_pdf[:-2] + 'ng'
    #plt.savefig(out_png, dpi=300)
    plt.close()


################################################################################
# filter_motif
#
# Collapse the filter parameter matrix to a single DNA motif.
#
# Input
#  param_matrix: np.array of the filter's parameter matrix
#  out_pdf:
################################################################################
def filter_motif(param_matrix):
    nts = 'ATCG'

    motif_list = []
    for v in range(param_matrix.shape[1]):
        max_n = 0
        for n in range(1,4):
            if param_matrix[n,v] > param_matrix[max_n,v]:
                max_n = n

        if param_matrix[max_n,v] > 0:
            motif_list.append(nts[max_n])
        else:
            motif_list.append('N')

    return ''.join(motif_list)


def plot_score_density(f_scores, out_pdf):
    sns.set(font_scale=1.3)
    plt.figure()
    sns.distplot(f_scores, kde=False)
    plt.xlabel('ReLU output')
    plt.savefig(out_pdf)
    plt.close()

    return f_scores.mean(), f_scores.std()
