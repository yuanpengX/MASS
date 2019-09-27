import os

import sys

ins = sys.argv[1]

fp1 = open(ins+'.fa','w')
count = 0
with open(ins,'r') as fp:
    for line in fp.readlines():
        fp1.write(f'>{count}\n')
        fp1.write(line.strip()+'\n')
        count+=1
        
