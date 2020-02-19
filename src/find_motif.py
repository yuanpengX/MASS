import sys
import os
from motif import *
#dirs = '/home/xiongyuanpeng/Research/m6a_prediction/motif/weblogo/mass/%s/filter%03d_logo/'
#model = 'mass-spe'
model = 'mass-single'
dirs = f'/data/group/m6A/{model}_share_conv_saliency/'
#dirs = '/data/group/m6A/mass/meme_motifs'
# f'/data/group/m6A/share_saliency_motifs/'#{model}/%s/filter%03d_logo/'

motif_dir = '/home/xiongyuanpeng/motif_databases/CISBP-RNA/'

sample_names =  ['hg19','panTro4','rheMac8','mm10','rn5','susScr3','danRer10',]
'''
meme_dic = {'hg19':f'{motif_dir}/HUMAN/HOCOMOCOv11_full_HUMAN_mono_meme_format.meme {motif_dir}/JASPAR/JASPAR2018_CORE_non-redundant.meme',\
'panTro4':f'{motif_dir}/JASPAR/JASPAR2018_CORE_non-redundant.meme',\
'rheMac8':f'{motif_dir}/JASPAR/JASPAR2018_CORE_non-redundant.meme',\
'mm10':f'{motif_dir}/MOUSE/HOCOMOCOv11_full_MOUSE_mono_meme_format.meme {motif_dir}/JASPAR/JASPAR2018_CORE_non-redundant.meme',\
'rn5':f'{motif_dir}/MOUSE/HOCOMOCOv11_full_MOUSE_mono_meme_format.meme {motif_dir}/JASPAR/JASPAR2018_CORE_non-redundant.meme',\
'susScr3':f'{motif_dir}/JASPAR/JASPAR2018_CORE_non-redundant.meme',\
'danRer10':f'{motif_dir}/JASPAR/JASPAR2018_CORE_non-redundant.meme',\
}
'''


meme_dic = {'hg19':'Homo_sapiens.meme',\
'panTro4':'Pan_troglodytes.meme',\
'rheMac8':'Macaca_mulatta.meme',\
'mm10':'Mus_musculus.meme',\
'rn5':'Rattus_norvegicus.meme',\
'susScr3':'Sus_scrofa.meme',\
'danRer10':'Danio_rerio.meme',\
}


id = sample_names[int(sys.argv[1])]
print(id)
for i in range(1): #range(len(sample_names)):
    sample = id#sample_names[i]
    #id = sample
    if True:#for i in range(32):
        dirname = dirs
        os.system(f'mkdir {dirname}/{sample}-tomtom/')
        cmd = 'tomtom  -no-ssc -oc %s  -verbosity 1 -min-overlap 5 -dist pearson -evalue -thresh 10.0 %s/%s.meme  %s/%s'%(dirname+ f'/{sample}-tomtom', dirname, sample, motif_dir, meme_dic[sample])
        print(cmd)
        os.system(cmd)
        #exit(o)
'''


for i in range(len(sample_names)):
    
    sample = sample_names[i]
    print(sample)
    num_filters = 32

    table_out = open('%s/table.txt'%(f'/data/group/m6A/{model}/%s/'%sample), 'w')
    #table_out = open('%s/table.txt'%('/home/xiongyuanpeng/Research/m6a_prediction/motif/weblogo/mass/%s/'%sample), 'w')
    # print header for later panda reading
    header_cols = ('', 'consensus', 'annotation', 'ic', 'p-value','e-value','q-value')
    table_out.write('%19s  %19s  %10s  %5s  %6s  %6s  %6s\n' % header_cols)
    for i in range(num_filters):
        dirname = dirs%(sample,i) 
        filter_names = name_filters(num_filters, '%s/tomtom/tomtom.tsv'%dirname, '/home/xiongyuanpeng/motif_databases/CISBP-RNA/' + meme_dic[sample])


        #################################################################
        # print a table of information
        #################################################################


        #for f in range(num_filters):
        # collapse to a consensus motif
        #consensus = filter_motif(filter_weights[f,:,:])
        consensus = 'ATCG'
        # grab annotation
        annotation = '.'
        pval = '1e14'
        evals = '1e14'
        qval = '1e14'
        name_pieces = filter_names.split('_')
        if len(name_pieces) > 1:
            annotation = name_pieces[1]
            pval = name_pieces[2]
            evals = name_pieces[3]
            qval = name_pieces[4]
        # plot density of filter output scores
        #fmean, fstd = plot_score_density(np.ravel(filter_outs[:,f,:]), '%s/filter%d_dens.pdf' % (options.out_dir,f))

        row_cols = ('filter_%d'%i, consensus, annotation, 0, pval, evals, qval)
        table_out.write('%19s  %19s  %10s  %5s  %6s  %6s %6s\n' % row_cols)

    table_out.close()

'''
'''

        #################################################################
        # global filter plots
        #################################################################
        if options.plot_heats:
            # plot filter-sequence heatmap
            plot_filter_seq_heat(filter_outs, '%s/filter_seqs.pdf'%options.out_dir)

            # plot filter-segment heatmap
            plot_filter_seg_heat(filter_outs, '%s/filter_segs.pdf'%options.out_dir)
            plot_filter_seg_heat(filter_outs, '%s/filter_segs_raw.pdf'%options.out_dir, whiten=False)

            # plot filter-target correlation heatmap
            plot_target_corr(filter_outs, seq_targets, filter_names, target_names, '%s/filter_target_cors_mean.pdf'%options.out_dir, 'mean')
            plot_target_corr(filter_outs, seq_targets, filter_names, target_names, '%s/filter_target_cors_max.pdf'%options.out_dir, 'max')
'''
