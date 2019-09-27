import os
from glob import glob
import random
basedir = '/home/xiongyuanpeng/Research/m6a_prediction/data/sequence_samples_downsample/'

for c in range(5):
    pos = glob(basedir+'/%d/positive_samples/valid/*_100'%c)
    print(pos)
    for name in pos:
        l = len(open(name).readlines())
        sample_size = l * 10
        lines = open((basedir +'/%d/negative_samples/valid/'%c + name.split('/')[-1].replace('pos','neg')),'r').readlines()
        #@print(name)
        #print(l)
        #print(len(lines))
        #random.shuffle(lines)
        
        lines_new = random.sample(lines, min(sample_size, len(lines)))
        
        fp = open((basedir +'/%d/negative_samples/valid/'%c + name.split('/')[-1].replace('pos','neg') + '_downsample.fa'),'w')
        count = 0
        for line in lines_new:
            fp.write('>count%d\n'%count)
            fp.write(line.strip() +'\n')
            count+=1
        
