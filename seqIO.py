# encoding: utf-8
# author: xiongyuanpeng
# 2018-11-15
import sklearn 
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import datetime
import random
from config import *
from util import *

class SequencesProvider:
    '''
    seq io provider
    '''
    # this order is very important
    
    def __init__(self, pos_dir, neg_dir, load_train = False):
        self.train_suffix = config.train_suffix#'_1000_100'#'_100_sramp_down'#
        self.val_suffix =config.val_suffix#'_1000_100_self_processed'# '_100_sramp_down'#
        self.test_suffix = config.test_suffix#self.val_suffix#'_100_sramp'

        self.sample_names = config.sample_names
        self.percent = 0# config.IO_STEP
        self.pos_dir = pos_dir
        self.neg_dir = neg_dir
        
        self.train_sample_index = {}
        self.valid_sample_index = {} 
        
        self.test_samples = {}
        self.test_labels = {}
        self.test_size_dic = {}
        self.test_sample_index = {}
        for i in range(config.TRAIN.CLASSES - 1):
            sample = self.sample_names[i]
            self.train_sample_index[sample] = [0,0]
            self.valid_sample_index[sample] = [0,0]
            self.test_sample_index[sample] = [0,0]
            self.test_samples[sample] = []
            self.test_labels[sample] = []
            self.test_size_dic[sample] = 0
            #self.test_sample_index[sample] = 0
        # train datasets
        self.pos_lists = {}
        self.neg_lists = {}
        self.pos_percent = 0#config.IO_STEP      
        self.neg_percent = 0
        self.train_size = 0#xffffff
        #
        self.valid_pos = {}
        self.valid_neg = {}
        self.valid_size = 0
        
        # test dataset
        self.test_pos = {}
        self.test_neg = {}
        self.test_size = 0
        if load_train: 
            print('[INFO] data_loader is loading valid data')
            self._load_valid()        
        if not load_train:
            print('[INFO] data_loader is loading test data')
            self._load_test() 

        # = len(self.test_samples)
        if load_train:
            print('[INFO] data_loader is loading train data')
            time = datetime.datetime.now()
            self._diskIO(update_pos = True, update_neg = True)
            print(datetime.datetime.now() - time)
    
    def _load_test_old(self):

        pos_dir = self.pos_dir +'/test/'
        neg_dir = self.neg_dir + '/test/'
        for i in range(config.TRAIN.CLASSES - 1):
            sample = self.sample_names[i]
            with open(neg_dir + '%s_neg_seqs'%(sample) + self.val_suffix, 'r') as fp:
                lines = fp.readlines()
                self.test_samples[sample].extend(lines)
                           
                self.test_labels[sample].extend([config.TRAIN.CLASSES - 1 ,]*len(lines))

            with open(pos_dir + '%s_pos_seqs'%(sample) + self.val_suffix, 'r') as fp:
                lines = fp.readlines()
                self.test_samples[sample].extend(lines)
              
                self.test_labels[sample].extend([i,]*len(lines))
            self.test_size_dic[sample] = len(self.test_labels[sample])
            
    def _load_test(self):

        pos_dir = self.pos_dir +'/test/'
        neg_dir = self.neg_dir + '/test/'
        for i in range(config.TRAIN.CLASSES - 1):
            sample = self.sample_names[i]
            with open(neg_dir + '%s_neg_seqs'%(sample) + self.test_suffix, 'r') as fp:
                self.test_neg[sample] = fp.readlines()
                #self.valid_size = max(self.valid_size, len(self.valid_neg[sample]))
            
            with open(pos_dir + '%s_pos_seqs'%(sample) + self.test_suffix, 'r') as fp:
                self.test_pos[sample] = fp.readlines()
                self.test_size = max(self.test_size, len(self.test_pos[sample]))

    def _load_valid(self):

        pos_dir = self.pos_dir +'/valid/'
        neg_dir = self.neg_dir + '/valid/'
        for i in range(config.TRAIN.CLASSES - 1):
            sample = self.sample_names[i]
            with open(neg_dir + '%s_neg_seqs'%(sample) + self.val_suffix, 'r') as fp:
                self.valid_neg[sample] = fp.readlines()
                #self.valid_size = max(self.valid_size, len(self.valid_neg[sample]))
            
            with open(pos_dir + '%s_pos_seqs'%(sample) + self.val_suffix, 'r') as fp:
                self.valid_pos[sample] = fp.readlines()
                self.valid_size = max(self.valid_size, len(self.valid_pos[sample]))
        
    def _diskIO(self, update_pos = False, update_neg = False):
        '''
        load samples in disks
        low, execute for several epochss
        pos should eg more frequen
        if cost near 20 min
        this function only used for rain data usage testdata litti ad into memorys
        '''
        # handling samples
        if update_pos:
            print('[INFO] Loading pos from disk!')     
        if update_neg:
            print('[INFO] Loading neg from disk!')
        pos_dir = self.pos_dir +'/train/'
        neg_dir = self.neg_dir + '/train/'
        
        for i in range(config.TRAIN.CLASSES - 1):
            sample = self.sample_names[i]
            def readfile(filename, start, end, SAMPLE=False):
                with open(filename, 'r') as fp:
                    lines = fp.readlines()
                    if SAMPLE:
                        return random.sample(lines, min(len(lines),config.DEBUG_SAMPLE_SIZE))
                    size = len(lines)                    
                    return lines[int(start * size):int(end * size)]
            def readall(filename):
                with open(filename, 'r') as fp:
                    return fp.readlines()

            if update_pos:#config.DEBUG:
                start = self.pos_percent        
                end = (self.pos_percent + config.POS_IO_STEP)            

                #tmp = readfile(pos_dir + '%s_pos_seqs_1000' % sample, start, end)
                #length = len(tmp)
                if config.DEBUG:
                    self.pos_lists[sample] = readfile(pos_dir + '%s_pos_seqs' % sample  + self.train_suffix, start, end, SAMPLE=True)#readall(pos_dir + '%s_pos_seqs_1000'%sample)#random.sample(tmp, min(length, 10000))
                
                else:
                    self.pos_lists[sample] = readfile(pos_dir + '%s_pos_seqs' % sample + self.train_suffix, start, end)

                self.train_size = max(self.train_size, len(self.pos_lists[sample]))
            if update_neg: # net allows different frequency with pos
                #print('[INFO] Loading neg from disk!')
                start = self.neg_percent
                end = (self.neg_percent + config.NEG_IO_STEP)

                if config.DEBUG:
                    
                    #tmp = readfile(neg_dir + '%s_neg_seqs_1000'%sample, start, end)
                    #length = len(tmp)
                    #self.neg_lists[sample] = random.sample(tmp, min(length, 10000))
 
                    self.neg_lists[sample] = readfile(neg_dir + '%s_neg_seqs'%sample + self.train_suffix, start, end, SAMPLE = True)
                else:
                    self.neg_lists[sample] = readfile(neg_dir + '%s_neg_seqs'%sample + self.train_suffix, start, end)
        np.random.shuffle(self.neg_lists[sample]) 
        # better for trainning
        # extra operation, every
        #self.train_size = max(self.train_size, len(self.neg_lists[sample]))
        self.pos_percent += config.POS_IO_STEP
        self.neg_percent += config.NEG_IO_STEP
        if self.pos_percent >= 1:
            self.pos_percent = 0
        if self.neg_percent >= 1:
            self.neg_percent = 0
    
    def _get_batch(self, pos_lists, neg_lists):
        
        sample_lines = []        
        
        labels  = []
        i = 0
        index = random.randint(0, config.TRAIN.CLASSES - 2)
        for j in range(config.SAMPLE):
            for i in range(config.TRAIN.CLASSES - 1):
                sample = self.sample_names[i]
                sample_lines.extend(random.sample(pos_lists[sample], 1))
                labels.append(i)
                if i == index:
                    sample_lines.extend(random.sample(neg_lists[sample], 1))
                    labels.append(config.TRAIN.CLASSES -  1)
        samples = readList(sample_lines)
        try:
            samples, labels = np.array(samples, np.float32), np.array(labels, np.int32)
            return samples, labels
        except:
            print('[Warning] get io error!')
            print(len(samples))
            print(len(labels))
            exit()
            return None, None   
            
    def get_next_valid_batch(self):
        samples, labels = None, None
        while samples is None:
            samples, labels =  self._get_batch(self.valid_pos, self.valid_neg)
        return samples, labels

    def get_next_train_batch(self):
        samples, labels = None, None
        samples, labels = None, None
        while samples is None:
            samples, labels =  self._get_batch(self.valid_pos, self.valid_neg)
        return samples, labels

    def get_next_train_batch(self):
        samples, labels = None, None
        while samples is None:
            samples, labels =  self._get_batch(self.pos_lists, self.neg_lists)
        return samples, labels
        
    def _get_next_batch_by_name(self, sample, pos_lists, neg_lists, sample_index):
        samples, labels = None, None
        BATCH = config.BATCH_SIZE
        while (samples is None):
            BATCH = config.BATCH_SIZE * config.NEG_POS_RATIO
            if sample_index[sample][0]  + BATCH > len(neg_lists[sample]):
                sample_index[sample][0] = 0
            start =  sample_index[sample][0]
            samples = neg_lists[sample][start:start + BATCH]#random.sample(neg_lists[sample], 5)            
            labels = [0,]* len(samples)
            

            BATCH = config.BATCH_SIZE
            if sample_index[sample][1]  + BATCH > len(pos_lists[sample]):
                sample_index[sample][1] = 0
            start =  sample_index[sample][1]
            
            sample_tmp = pos_lists[sample][start:start+BATCH]
            samples.extend(sample_tmp)#(random.sample(pos_lists[sample], 5))
            labels.extend([1,]* len(sample_tmp))
            
            samples = readList(samples)
            
            sample_index[sample][0] +=BATCH * config.NEG_POS_RATIO
            sample_index[sample][1] +=BATCH
            
            try:
                samples, labels = np.array(samples, np.float32), np.array(labels, np.int32)
                if len(samples) > 0:
                
                    return samples, labels, sample_index
                else:
                    samples, lables = None
            except:
                samples, labels = None, None
                
    def get_next_train_batch_by_name(self, sample):
        samples, labels, self.train_sample_index = self._get_next_batch_by_name(sample, self.pos_lists, self.neg_lists, self.train_sample_index)
        return samples, labels
        
    def get_next_valid_batch_by_name(self, sample):
        samples,labels,self.valid_sample_index=  self._get_next_batch_by_name(sample, self.valid_pos, self.valid_neg, self.valid_sample_index)
        return samples, labels
    
    def get_next_test_batch_by_name(self, sample):
        samples,labels,self.test_sample_index=  self._get_next_batch_by_name(sample, self.test_pos, self.test_neg, self.test_sample_index)
        return samples, labels
    
    def get_next_test_batch_by_name_old(self, sample):
        
        #self.test_sample_index[sample]
        samples, labels = None, None
        while samples is None:
            start = self.test_sample_index[sample]
            end = start + config.BATCH_SIZE * 3
            
            samples = self.test_samples[sample][start: end]
            labels = self.test_labels[sample][start: end]
            
            samples = readList(samples)
            try:
                samples, labels = np.array(samples, np.float32), np.array(labels, np.int32)
                if len(samples) > 0:
                    return samples, labels
                else:
                    samples, lables = None, None
            except:
                samples, labels = None, None
            start = end
            if start >= self.test_size[sample]:
                start = 0
            self.test_sample_index[sample] =  start
        #return samples, labels


    def get_next_test_batch(self):
        '''
        This part is quiet different from test and valid(no sampling here!)
        '''
        while True:
            start = self.test_index
            end = min(self.test_index + config.BATCH_SIZE, self.test_size -1)
            self.test_index +=config.BATCH_SIZE
            
            if self.test_index > self.test_size -1:
                self.test_index = 0
            samples = readList(self.test_samples[start:end])
            labels = self.test_labels[start:end]
            try:
                samples, labels = np.array(samples, np.float32), np.array(labels,np.int32)
                return samples, labels 
            except:
                print('\n[Warning] get io error!')

    def get_test_size_by_name(self, sample):
        return self.test_size_dic[sample]
    
    def get_test_size(self):
        return self.test_size
    def get_valid_size(self):
        return self.valid_size
 
    def get_train_size(self):
        return self.train_size

        
class SequencesProvider2:
    '''
    seq io provider
    '''
    # this order is very important
    sample_names = config.sample_names
    
    percent = 0# config.IO_STEP
    def __init__(self, pos_dir, neg_dir, load_train = False):

        self.pos_dir = pos_dir
        self.neg_dir = neg_dir
        
        # handling files index
        self.pos_percent = np.zeros(config.TRAIN.CLASSES - 1)
        self.neg_percent = np.zeros(config.TRAIN.CLASSES - 1)
        
        self.train_samples = []
        self.train_labels = []
        self.train_size = 0
        
        # handling training samples index
        self.pos_start = 0
        self.neg_start = 0
        
        self.test_samples =[]
        self.test_labels  =[]
        self.test_index = 0
        
        
        self.valid_neg = {}
        self.valid_pos = {}
        self.valid_size= 0
        
        print('[INFO] data_loader is loading valid data')
        self._load_valid()        
        print('[INFO] data_loader is loading test data')
        self._load_test() 
        self.test_size = len(self.test_samples)
        if load_train:        
            print('[INFO] data_loader is loading train data')
            self._diskIO()
        
    def _load_test(self):

        pos_dir = self.pos_dir +'/test/'
        neg_dir = self.neg_dir + '/test/'
        for i in range(config.TRAIN.CLASSES - 1):
            sample = self.sample_names[i]
            with open(neg_dir + '%s_neg_seqs_1000'%(sample), 'r') as fp:
                lines = fp.readlines()
                self.test_samples.extend(lines)
                           
                self.test_labels.extend([config.TRAIN.CLASSES - 1 ,]*len(lines))

            with open(pos_dir + '%s_pos_seqs_1000'%(sample), 'r') as fp:
                lines = fp.readlines()
                self.test_samples.extend(lines)
              
                self.test_labels.extend([i,]*len(lines))
                
    def _load_valid(self):
        pos_dir = self.pos_dir +'/valid/'
        neg_dir = self.neg_dir + '/valid/'
        for i in range(config.TRAIN.CLASSES - 1):
            sample = self.sample_names[i]
            with open(neg_dir + '%s_neg_seqs_1000'%(sample), 'r') as fp:                                
                self.valid_neg[sample] = fp.readlines()
                self.valid_size = max(self.valid_size, len(self.valid_neg[sample]))
            
            with open(pos_dir + '%s_pos_seqs_1000'%(sample), 'r') as fp:
                self.valid_pos[sample] = fp.readlines()
                self.valid_size = max(self.valid_size, len(self.valid_pos[sample]))
        
    def _diskIO(self, update_pos = True, update_neg =True):
        
        # every time we load parts of samples
        # pos:  1/10 percent
        # neg:  (1/100 percent, 20000) 
        pos_dir = self.pos_dir +'/train/'
        neg_dir = self.neg_dir + '/train/'
        
        def read(base_dir, step, percent):
            samples = []
            labels = []
            for i in range(config.TRAIN.CLASSES - 1):
                sample = self.sample_names[i]
                with open(base_dir % sample) as fp:
                    lines = [line for line in fp.readlines() if len(line.strip()) > 0]
                    start = int(percent[i])
                    end = start + int(step * len(lines))
                    if (end - start) > 30000:
                        end = start + 30000
                    if end > len(lines):
                        percent[i] = 0
                    else:
                        percent[i] = end
                    lines = lines[start:end]
                samples.extend(lines)
                labels.extend([i,]* len(lines))
            index = np.arange(0,len(samples))
            random.shuffle(index)
            
            return [samples[i] for i in index], [labels[i] for i in index], percent
        
        
        self.pos_samples, self.pos_labels, self.pos_percent = read(pos_dir + '%s_pos_seqs' + self.train_suffix,0.1, self.pos_percent)
        self.train_size = len(self.pos_samples)
        
        self.neg_samples, self.neg_labels, self.neg_percent = read(neg_dir + '%s_neg_seqs'  + self.train_suffix,0.01, self.neg_percent)
        
    def _get_batch(self, pos_lists, neg_lists):
        
        sample_lines = []        
      
        labels  = []
        i = 0
        index = random.randint(0, config.TRAIN.CLASSES - 2)
        for j in range(config.SAMPLE):
            for i in range(config.TRAIN.CLASSES - 1):
                sample = self.sample_names[i]
                sample_lines.extend(random.sample(pos_lists[sample], 1))
                labels.append(i)
                if i == index:
                    sample_lines.extend(random.sample(neg_lists[sample], 1))
                    labels.append(config.TRAIN.CLASSES -  1)
        samples = readList(sample_lines)
        try:
            samples, labels = np.array(samples, np.float32), np.array(labels, np.int32)
            return samples, labels
        except:
            print('[Warning] get io error!')
            print(len(samples))
            print(len(labels))
            exit()
            return None, None   
            
    def get_next_valid_batch(self):
        samples, labels = None, None
        while samples is None:
            samples, labels =  self._get_batch(self.valid_pos, self.valid_neg)
        return samples, labels

    def get_next_train_batch(self):
        while True:
            # half of them are neg samples
            start = self.pos_start 
            end = start + config.SAMPLE * (config.TRAIN.CLASSES - 1)
            samples = self.pos_samples[start:end]
            labels = self.pos_labels[start:end]
            self.pos_start = end
            if self.pos_start > len(self.pos_labels):
                self.pos_start = 0
                
            start = self.neg_start
            end = start + config.SAMPLE
            self.neg_start = end
            if self.neg_start> len(self.neg_labels):
                self.neg_start = 0
            samples.extend(self.neg_samples[start:end])
            labels.extend(self.neg_labels[start:end])
            samples = readList(samples)
            
            try:
                samples, labels = np.array(samples, np.float32), np.array(labels,np.int32)
                return samples, labels 
            except:
                print('\n[Warning] get io error!')
        
    def get_next_test_batch(self):
        '''
        This part is quiet different from test and valid(no sampling here!)
        '''
        while True:
            start = self.test_index
            end = min(self.test_index + config.BATCH_SIZE, self.test_size -1)
            self.test_index +=config.BATCH_SIZE
            
            if self.test_index > self.test_size -1:
                self.test_index = 0
            samples = readList(self.test_samples[start:end])
            labels = self.test_labels[start:end]
            try:
                samples, labels = np.array(samples, np.float32), np.array(labels,np.int32)
                return samples, labels 
            except:
                print('\n[Warning] get io error!')
        
    def get_test_size(self):
        return self.test_size
        
    def get_train_size(self):
        return self.train_size
        
    def get_valid_size(self):
        return self.valid_size
        
        
if __name__ == "__main__":
    import datetime
    start = datetime.datetime.now()
    sp = SequencesProvider2( '../data/sequence_samples/positive_samples/', '../data/sequence_samples/negative_samples/', load_train = False)
    print(sp.get_train_size())
    print(sp.get_test_size())
    print(datetime.datetime.now() - start)
