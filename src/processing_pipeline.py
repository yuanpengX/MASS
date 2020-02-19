import os
from glob import glob
import random
def cat1000to100(basedir):
    files = glob(basedir)
    
    for f in files:
        print('processing %s'%f)
        with open(f,'r') as fp:
            fp1 = open(f+'_100','w')
            for line in fp.readlines():
                line = line.strip()[450:-450]
                fp1.write(line+'\n')

def downsample_neg(basedir):
    names = glob(basedir)    
    #pos = glob(basedir+'/%d/positive_samples/valid/*_100'%c)
    if True:
        for name in names:
            print('processing %s'%name)
            l = len(open(name).readlines())
            sample_size = l * 10
            neg_name = name.replace('positive_samples','negative_samples').replace('pos','neg')
            
            lines = open(neg_name,'r').readlines()
            #@print(name)
            #print(l)
            #print(len(lines))
            #random.shuffle(lines)
            
            lines_new = random.sample(lines, min(sample_size, len(lines)))
            
            fp = open(neg_name + '.fa','w')
            count = 0
            for line in lines_new:
                fp.write('>count%d\n'%count)
                fp.write(line.strip() +'\n')
                count+=1

def conv_data2fa(base_dir):
    names = glob(base_dir)
    for name in names:
        with open(name, 'r') as fp:
            fp1  = open(name  + '.fa','w')
            count = 0
            for line in fp.readlines():
                line = line.strip()
                fp1.write('>seq%d\n'%count)
                fp1.write(line+'\n')
                count+=1
            fp1.close()
        
def runCDHIT(basedir):
    for name in glob(basedir):
        
        print('processing %s'%name)
        '''
            if not (name.startswith(sp)):
            continue
        '''
        #cmd = '/home/xiongyuanpeng/cdhit-master/cd-hit-est-2d -i %s -i2 %s -o %s -c 0.8 -n 5 -M 44810'%(basedir + l + 'train/' + name + '.fa', basedir + l + 'test/' + name + '.fa', basedir + l + 'test/' + name + '_75_processed.fa')

        cmd = '/home/xiongyuanpeng/cdhit-master/cd-hit-est -i %s -o %s -c 0.8 -n 4&'%(name ,  name.split('.')[0] + '_self_processed.fa')
        print(cmd)
        os.system(cmd)

def fa2data(basedir):
    
    names = glob(basedir)

    for name in names:
        print('processsing ', name)
        count = 1
        with open(name,'r') as fp:
            fp1 = open(name.split('.')[0],'w')
            for line in fp.readlines():
                if count % 2 == 0:
                    #print(line.strip())
                    fp1.write(line.strip()+'\n')
                count +=1
                #print(count)

def remove_fa():
    os.system('yesiwantremove ')
# processing validation set
print('Doing cat seqences')
'''
cat1000to100('/home/xiongyuanpeng/Research/m6a_prediction/data/sequence_samples/*/*/*_1000')
downsample_neg('/home/xiongyuanpeng/Research/m6a_prediction/data/sequence_samples/positive_samples/valid/*_100')
downsample_neg('/home/xiongyuanpeng/Research/m6a_prediction/data/sequence_samples/positive_samples/test/*_100')

conv_data2fa('/home/xiongyuanpeng/Research/m6a_prediction/data/sequence_samples/positive_samples/valid/*_100')
conv_data2fa('/home/xiongyuanpeng/Research/m6a_prediction/data/sequence_samples/positive_samples/test/*_100')

runCDHIT('/home/xiongyuanpeng/Research/m6a_prediction/data/sequence_samples/*/valid/*_100.fa')
runCDHIT('/home/xiongyuanpeng/Research/m6a_prediction/data/sequence_samples/*/test/*_100.fa')
'''
#fa2data('/home/xiongyuanpeng/Research/m6a_prediction/data/sequence_samples_downsample/*/*/*/*_100_self_processed.fa')
fa2data('/home/xiongyuanpeng/Research/m6a_prediction/data/sequence_samples/negative_samples/valid/mm10_neg_seqs_1000_100.fa')
 

