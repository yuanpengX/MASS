import sys
import numpy
import random
with open(sys.argv[1],'r') as fp:
    fp1 = open(sys.argv[1]+'_down','w')
    fp2 = open('../valid/'+sys.argv[1]+'_down','w')
    lines = fp.readlines()
    random.shuffle(lines)
    train_size = int((1-1/8)*len(lines))
    train_lines = lines[:train_size]
    valid_lines = lines[train_size:]
    
    def write_lines(fp, lines):
        for line in lines:
            fp.write(line.strip()+'\n')
    write_lines(fp1, train_lines)
    write_lines(fp2, valid_lines)

    
