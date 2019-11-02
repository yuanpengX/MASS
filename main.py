# encoding: utf-8
# author: xiongyuanpeng
# 2018-11-14
#
import argparse
from model import *
import datetime
from seqIO import *
import sys
from sklearn import metrics
import pandas as pd
import numpy as np
import math
from sklearn.metrics import precision_score, recall_score, f1_score, precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

from tensorflow.contrib.layers import l2_regularizer as l2reg
model_dict  = {'sharedFeatureExtractor': sharedFeatureExtractor,'sharedFeatureExtractor2': sharedFeatureExtractor2, 
            'DeepM6ASeq_pre':DeepM6ASeq_pre, 'sharedFeatureExtractor3': sharedFeatureExtractor3,'AttentionSeqs':AttentionSeqs,
            'sharedFeatureExtractor_nodropout':sharedFeatureExtractor_nodropout}
model_dict2 = {'classifier':classifier,'DeepM6ASeq':DeepM6ASeq,'classifierSequences':classifierSequences}

np.random.seed(65535)  # for reproductibility

class M6ANet():
    
    def __init__(self, data_loader, model_name):
        
        self.data_loader = data_loader
        self.checkpoint_dir = config.CHECK_POINT_DIR
        # define inputs and target
        self.t_sequences = tf.placeholder(tf.float32, [None, config.TRAIN.TIME_STEPS, config.TRAIN.EMBED_DIM])
        self.t_target = tf.placeholder(tf.int32, [None, ])
        
        # define module
        print(datetime.datetime.now())
        print('generating selector')
        print(datetime.datetime.now())
        self.selecting, self.selecting_logits = eval(model_name + '(self.t_sequences)')
        print('generating predictor')
        print(datetime.datetime.now())
        self.category, self.probs = Predictor(self.selecting, self.t_sequences)
        
        # get parameters
        #self.selector_vars = tl.layers.get_variables_with_name('selector')
        #self.predictor_vars = tl.layers.get_variables_with_name('predictor')
        # merge variable lists
        #self.all_vars = self.selector_vars + self.predictor_vars
        
        # define loss
        self.entrop_loss = tl.cost.cross_entropy(self.category.outputs, self.t_target, name = 'entrop')
        self.l1_loss = tf.reduce_mean(tf.abs(self.selecting_logits.outputs))
        self.loss = self.entrop_loss + config.TRAIN.LAMBDA * self.l1_loss
        # set optimizer
        self.pre = 0
        print('generating all optimizer')
        print(datetime.datetime.now())
        self.all_optim = tf.train.AdamOptimizer(config.TRAIN.INIT_LR, beta1=config.TRAIN.beta1).minimize(self.loss, var_list = self.category.all_params)
        print(datetime.datetime.now())
        #print('generating selector optimizer')
        #self.selector_optim = tf.train.AdamOptimizer(config.TRAIN.INIT_LR, beta1=config.TRAIN.beta1).minimize(self.entrop_loss, var_list=self.selector_vars)
        #print('generating predictor optimizer')
        #self.predictor_optim = tf.train.AdamOptimizer(config.TRAIN.INIT_LR, beta1=config.TRAIN.beta1).minimize(self.entrop_loss, var_list=self.predictor_vars)
        
        # get session and initialize
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
        tl.layers.initialize_global_variables(self.sess)
        self.load() 
    def load(self):
        '''
        directly load from chekpoint dir
        '''
        name = self.checkpoint_dir + '/m6aNet_{}.npz'.format(tl.global_flag['name'])
        if os.path.exists(name):
            tl.files.load_and_assign_npz(sess=self.sess, name=name, network=self.category)
    
    def save(self, auc):
        if self.pre < auc:
            self.pre = auc
        else:
            return
        # save file into checkpoint dir
        tl.files.save_npz(self.category.all_params, name=self.checkpoint_dir + '/m6aNet_{}.npz'.format(tl.global_flag['name']), sess=self.sess)
    
    def train(self):
        
        for epoch in range(config.TRAIN.MAX_EPOCHS):
            #batch_size = config.TRAIN.CLASSES * 2 * config.SAMPLE
            max_iter = int(self.data_loader.get_train_size() / config.BATCH_SIZE)
            EL = 0
            #AUC = 0
            ACC = 0
            #print(max_iter)
            #exit() 
            for idx in range(max_iter):
                #print(datetime.datetime.now())
                seq, label = self.data_loader.get_next_train_batch()
                #print(label)
                label_onehot = pd.get_dummies(label).values.tolist()
                #print(datetime.datetime.now())
                feed_dict = {self.t_sequences:seq, self.t_target:label}
                el,_, pred,select = self.sess.run([self.entrop_loss, self.all_optim, self.probs, self.selecting.outputs], feed_dict)
                pred_label = np.argmax(pred, axis = 1)
                '''
                acc = np.zeros(config.TRAIN.CLASSES)
                for i in range(len(label)):
                    acc[label[i]] += int(label[i] == pred_label[i])
                '''
                #print(acc)
                #acc = np.mean(acc)
                acc = np.sum(np.array(pred_label) == np.array(label)) / len(pred_label)
                ACC += acc#np.mean(acc)
                #print(datetime.datetime.now())
                #exit()
                EL +=el
                #print('label:',label)
                #print('predict:',pred_label)
                #print(np.argmax(select[:,:,0],axis=1))
                #exit()
                #auc = metrics.roc_auc_score(np.array(label_onehot), np.array(pred), average = 'micro')
                #AUC += auc
                if(idx % 100) == 0:
                    percent = (idx + 1) * 50 / max_iter
                    num_arrow = int(percent)
                    num_line = 50 - num_arrow
                    progress_bar = '[' + '>' * num_arrow + '-' * num_line + ']' + '%.2f%% ' % (percent*2) + '  CrossEntrop: %.6e, acc = %.2f' % (el, acc) + '\r'
                    sys.stdout.write(progress_bar)
                    sys.stdout.flush()
            val_ce, val_auc, val_acc = self.evaluate()
            # evaluate model for every epochs
            print("\n[INFO] Epoch (%d/%d) train cross_entrop: %.3f acc : %.3f, valid cross_entrop: %.3f auc: %.3f acc: %.3f"%(epoch, config.TRAIN.MAX_EPOCHS, EL/max_iter, ACC / max_iter, val_ce, val_auc, val_acc))
            if True:#(epoch % config.TRAIN.SAVE_EVERY is 0) and (epoch is not 0):
                self.save(val_acc)
            if True:#not config.DEBUG:
                if (epoch is not 0) and (epoch % config.TRAIN.IO_POS_EVERY) is 0:
                    self.data_loader._diskIO()
                #if (epoch is not 0) and (epoch % config.TRAIN.IO_NEG_EVERY) is 0:
                #    self.data_loader._diskIO(update_neg = True)

    
    def evaluate(self):
        print('\n[INFO] evaluating:')
        batch_size = config.TRAIN.CLASSES * 2 * config.SAMPLE     
        max_iter = 500#int((data_loader.get_test_size() * (config.TRAIN.CLASSES - 1))/batch_size)
        EL = 0
        AUC = 0
        ACC = 0
        for idx in range(max_iter):
            seq, label = self.data_loader.get_next_valid_batch()
            label_onehot = pd.get_dummies(label).values.tolist() 
            feed_dict = {self.t_sequences:seq, self.t_target:label}
            el,pred = self.sess.run([self.entrop_loss, self.probs], feed_dict)
            pred_label = np.argmax(pred, axis = 1)
            #print('pred:', idx, pred_label)
            #print('label:',idx, label)
            if config.TRAIN.CLASSES == 2:
                print(label)
                print(pred_label)
                #print(precision_score(label,pred_label), recall_score(label, pred_label), f1_score(label, pred_label))
                print(np.sum(np.logical_and(np.equal(label,1), np.equal(pred_label, 0))))
                print(np.sum(np.logical_and(np.equal(label,0), np.equal(pred_label, 0))))
                #exit()
            acc = np.sum(pred_label == label) / len(pred_label)
            auc = metrics.roc_auc_score(np.array(label_onehot), np.array(pred), average = 'micro')
            AUC += auc
            ACC +=acc
            EL+=el
        return EL/max_iter, AUC/max_iter, ACC/max_iter
     
    def test(self):
        print('[INFO] testing:')

        batch_size = config.BATCH_SIZE
        AUC = 0
        max_iter = math.ceil(self.data_loader.get_test_size()/batch_size)
        ACC = 0
        for idx in range(max_iter):
            seq, label = self.data_loader.get_next_test_batch()
            label_onehot = pd.get_dummies(label).values.tolist()
            feed_dict = {self.t_sequences:seq, self.t_target:label}
            el,pred = self.sess.run([self.entrop_loss, self.probs], feed_dict)
            pred_label = np.argmax(pred, axis=1)

            #auc = metrics.roc_auc_score(np.array(label_onehot), np.array(pred), average = 'micro')
            #AUC+=auc
            #print(pred)
            #print(pred_label)
            #print(label)
            
            acc = np.sum(pred_label == label)       

            if(acc < 0.7 * config.BATCH_SIZE):
                print(pred)
                print(pred_label) 
                print(label) 
                exit()
            #print(acc)
            #exit()
            ACC +=acc
            percent = (idx + 1) * 50 / max_iter
            num_arrow = int(percent)
            num_line = 50 - num_arrow
            progress_bar = '[' + '>' * num_arrow + '-' * num_line + ']' + '%.2f%% ' % (percent*2) + ('acc is %.2f'%(acc/config.BATCH_SIZE)) + '\r'
            sys.stdout.write(progress_bar)
            sys.stdout.flush()
            #val_ce, val_auc = self.evaluate()
        print('AUC is %.2f'%(AUC/max_iter))
        print('ACC is %.2f'%(ACC /self.data_loader.get_test_size()))

EMBEDDING = True
CLASS_SEQ = False
class M6ANetShare():
    sample_names = config.sample_names
    def __init__(self, data_loader, model1 = 'sharedFeatureExtractor2', model2 = 'classifier'):
        self.pre = 0
        self.data_loader = data_loader
        self.checkpoint_dir = config.CHECK_POINT_DIR
        self.extractor = model_dict[model1]
        self.classifier = model_dict2[model2]
        if EMBEDDING:
                self.t_sequences = tf.placeholder(tf.float32, [None, config.TRAIN.TIME_STEPS, config.TRAIN.EMBED_DIM])
        else:
            self.t_sequences = tf.placeholder(tf.int32, [None, config.TRAIN.TIME_STEPS])#, config.TRAIN.EMBED_DIM])
        # initialize t_targets groups
        self.targets = {}
        for id in range(config.TRAIN.CLASSES - 1):
            sample = self.sample_names[id]
            self.targets[sample] = tf.placeholder(tf.int32, [None,], name = sample + '_target')

        self.train_list = {}
        self.features, self.feature1 =  self.extractor(self.t_sequences, 'extractor', is_train = True)#eval(model1 + "(self.t_sequences, 'extractor', is_train = True)")
        
        self.classes_list = []
        for id in range(config.TRAIN.CLASSES - 1):
            sample = self.sample_names[id]
            if CLASS_SEQ:
                classes = self.classifier(self.features, self.t_sequences,sample, reuse = False, is_train = True)#eval(model2 + '(self.features, sample, reuse = False, is_train = True)')
            else:

                classes = self.classifier(self.features, sample, reuse = False, is_train = True)#eval(model2 + '(self.features, sample, reuse = False, is_train = True)')
            self.classes_list.append(classes)
        if config.DO_MULTI:
            ml_wraper = MultiWrapper(config.group_setting)
        
        for id in range(config.TRAIN.CLASSES - 1):
            sample = self.sample_names[id]
            classes = self.classes_list[id]
            if config.DO_MULTI:
                cost = tl.cost.cross_entropy(classes.outputs, self.targets[sample], name = sample + '_entrop') + ml_wraper.get_loss(sample)
            else:
                cost = tl.cost.cross_entropy(classes.outputs, self.targets[sample], name = sample + '_entrop') #+ config.TRAIN.L2LAMBDA * ()
            features_test,_ = self.extractor(self.t_sequences, 'extractor', reuse = True, is_train = False)# eval(model1 +"(self.t_sequences, 'extractor',reuse = True, is_train = False)")
            if CLASS_SEQ:
                classes_test = self.classifier(features_test, self.t_sequences,sample, reuse = True, is_train = False)#eval(model2 + '(features_test, sample, reuse = True, is_train = False)')
            else:
                classes_test = self.classifier(features_test, sample, reuse = True, is_train = False)#eval(model2 + '(features_test, sample, reuse = True, is_train = False)')

            #print(classes.all_params)
            #exit()
            optim_extractor = tf.train.AdamOptimizer(config.TRAIN.INIT_LR / 10, beta1=0.9, name =sample + '_adam_extractor').minimize(cost, var_list = tl.layers.get_variables_with_name('extractor', True, True))
            optim_all = tf.train.AdamOptimizer(config.TRAIN.INIT_LR, beta1=0.9, name =sample + '_adam_all').minimize(cost, var_list = classes.all_params)
            optim_one = tf.train.AdamOptimizer(config.TRAIN.INIT_LR, beta1=0.9, name =sample + '_adam_one').minimize(cost, var_list = tl.layers.get_variables_with_name(sample, True, True))
            if config.TRAIN.OPTIM_CLASSIFIER:
                optim = optim_one
            else:
                optim = optim_all
            self.train_list[sample] = [cost, optim_one, tf.nn.softmax(classes_test.outputs), optim_extractor]
           
        gpu_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        gpu_config.gpu_options.allow_growth = True
        
        self.sess = tf.Session(config=gpu_config)#tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
        tl.layers.initialize_global_variables(self.sess)

        self.load()
        #
    def train(self):
        
        for epoch in range(config.TRAIN.MAX_EPOCHS):

            max_iter = int(self.data_loader.get_train_size() / config.BATCH_SIZE)
            print('[INFO] EPOCH', epoch, ' size is ', max_iter)
            feed_dict = {self.t_sequences: np.random.randn(config.BATCH_SIZE * 2,config.TRAIN.TIME_STEPS, config.TRAIN.EMBED_DIM)}

            for id in range(config.TRAIN.CLASSES - 1):
                sample = self.sample_names[id]
                feed_dict[self.targets[sample]] = np.random.randint(0,2,size=config.BATCH_SIZE * 2)
            ACC_TRAIN = []
            for idx in range(max_iter):
                # at every iteration, we change samples
                LOSS = 0
                ACC = []
                AUC = []
                for id in range(config.TRAIN.CLASSES - 1):
                    sample = self.sample_names[id]
                    sequences, labels = self.data_loader.get_next_train_batch_by_name(sample)
                    feed_dict[self.t_sequences] = sequences
                    feed_dict[self.targets[sample]] = labels
                    label_onehot= pd.get_dummies(labels).values.tolist()
                    loss,_,probs,_ = self.sess.run(self.train_list[sample], feed_dict)
                    auc = metrics.roc_auc_score(np.array(label_onehot), np.array(probs), average = 'micro')
                    AUC.append(auc)
                    
                    LOSS += loss
                    pred_label = np.argmax(probs, axis =1)
                    acc = np.sum(np.array(pred_label) == np.array(labels)) / len(pred_label)
                    ACC.append(acc)
                LOSS = LOSS / len(self.sample_names)
                acc = np.mean(ACC)
                ACC_TRAIN.append(acc)
                mystr = ''
                aucstr = ''
                for id in range(len(ACC)):
                    mystr = mystr + ' %.2f '%(ACC[id])
                    aucstr = aucstr + ' %.2f '%(AUC[id])
                if True:#(idx % 100) == 0:
                    percent = (idx + 1) * 50 / max_iter
                    num_arrow = int(percent)
                    num_line = 50 - num_arrow
                    progress_bar = '[' + '>' * num_arrow + '-' * num_line + ']' + '%.2f%% ' % (percent*2) + '  CE: %.3e, acc = %.3f' % (LOSS, acc) + " sep " + mystr+ ' auc ' + aucstr + '\r'
                    sys.stdout.write(progress_bar)
                    sys.stdout.flush()
                    
            val_acc, val_auc = self.evaluate()
            mystr = ''
            aucstr = '0'
            #val_acc = [val_acc, ]
            #val_auc = [val_auc,]
            #print(type(val_acc))
            for id in range(len(val_acc)):
                mystr = mystr + ' %.2f '%(val_acc[id])
                aucstr = aucstr + ' %.2f '%(val_auc[id])
                
            print('\nEpoch %d/%d'%(epoch, config.TRAIN.MAX_EPOCHS),'train acc is %.3f accuracy is: %.3f'%(np.mean(ACC_TRAIN),np.mean(val_acc)), ' sep ' + mystr + ' auc ' + aucstr)
            self.save(np.mean(val_auc))

            if (epoch % config.TRAIN.IO_POS_EVERY) is 0:
                self.data_loader._diskIO(update_pos = True, update_neg = True)
                #if (epoch is not 0) and (epoch % config.TRAIN.IO_NEG_EVERY) is 0:
                #    self.data_loader._diskIO(update_pos = True, update_neg = True)
    def evaluate(self):
        print('\nEvaluating:')
        ACC_ALL = []
        AUC_ALL = []
        max_iter = int(self.data_loader.get_valid_size() / config.BATCH_SIZE)
         
        for idx in range(max_iter):
            feed_dict = {self.t_sequences: np.random.randn(config.BATCH_SIZE,config.TRAIN.TIME_STEPS, config.TRAIN.EMBED_DIM)}
            ACC = []
            AUC = []
            for id in range(config.TRAIN.CLASSES - 1):
                sample = self.sample_names[id]
                sequences, labels = self.data_loader.get_next_valid_batch_by_name(sample)
                feed_dict[self.t_sequences] = sequences
                feed_dict[self.targets[sample]] = labels
                label_onehot= pd.get_dummies(labels).values.tolist()
                probs = self.sess.run(self.train_list[sample][2], feed_dict)
                auc = metrics.roc_auc_score(np.array(label_onehot), np.array(probs), average = 'micro')
                pred_label = np.argmax(probs, axis =1)
                acc = np.sum(np.array(pred_label) == np.array(labels)) / len(pred_label)

                AUC.append(auc)
                ACC.append(acc)

            ACC_ALL.append(ACC)
            AUC_ALL.append(AUC)

        return np.mean(ACC_ALL, axis = 0), np.mean(AUC_ALL, axis = 0)

    def load(self):
        '''
        directly load from chekpoint dir
        '''
        import os
        name = self.checkpoint_dir + '/m6aNet_feature_{}.npz'.format(tl.global_flag['name'])
        print(name)
        if os.path.exists(name):
            tl.files.load_and_assign_npz(sess = self.sess, network = self.features, name = name)
        for i in range(len(self.classes_list)):
            sample = self.sample_names[i]
            name = self.checkpoint_dir + '/m6aNet_classese_{}_{}.npz'.format(sample, tl.global_flag['name'])
            if os.path.exists(name):
                tl.files.load_and_assign_npz(network = self.classes_list[i], name=name, sess=self.sess)


    def test_old(self):
        print('\nTesting:')
        ACC_ALL = []
        AUC_ALL = []
        
        for id in range(config.TRAIN.CLASSES - 1): 
            sample = self.sample_names[id]
            max_iter = math.ceil(self.data_loader.get_test_size_by_name(sample) / (config.BATCH_SIZE)*3)
            feed_dict = {self.t_sequences: np.random.randn(config.BATCH_SIZE,config.TRAIN.TIME_STEPS, config.TRAIN.EMBED_DIM)}
            
            probs_list = []
            labels_list = []
            
            for idx in range(max_iter):

                sequences, labels = self.data_loader.get_next_test_batch_by_name(sample)
                feed_dict[self.t_sequences] = sequences
                feed_dict[self.targets[sample]] = labels
                label_onehot= pd.get_dummies(labels).values.tolist()
                probs = self.sess.run(self.train_list[sample][2], feed_dict)
                #auc = metrics.roc_auc_score(np.array(label_onehot), np.array(probs), average = 'micro')
                #pred_label = np.argmax(probs, axis =1)
                #acc = np.sum(np.array(pred_label) == np.array(labels)) / len(pred_label)
                labels_list.extend(label_onehot)
                probs_list.extend(probs)
            print('Doing computing')
            # compute accuracy, precision_score, auc
            #label_onehot= pd.get_dummies(labels_list).values.tolist()
            prob = np.argmax(probs_list, axis = 1)
            print('Calculating AUC')
            auc = metrics.roc_auc_score(prob,labels_list, average = 'micro')
            print('Calculating F1 score')
            f1 = f1_score(prob,labels_list)
            print('Sample {} f1: {} auc: {}'.format(sample,f1, auc))
            
    def motif(self):
        print('calling motif')
        for idx in range(config.TRAIN.CLASSES - 1):
            sample = self.sample_names[idx]
            print('Processing:', sample)
            tl.files.exists_or_mkdir('../motif/weblogo/%s/'%sample)
            file_name = '/home/xiongyuanpeng/Research/m6a_prediction/data/sequence_samples/positive_samples/train/%s_pos_seqs_1000'%(sample)
            seqs = open(file_name).readlines()
            inputs = readList(seqs)
            inputs_np = np.array(inputs)
            filter_out = []
            test = np.random.randint(0,2,size= 16)
            feed_dict = {self.t_sequences: np.random.randn(config.BATCH_SIZE,config.TRAIN.TIME_STEPS, config.TRAIN.EMBED_DIM)}
            print('running output')
            for i in range(int(inputs_np.shape[0]/16)-1):
                input_np = inputs_np[i*16:min((i+1)*16,inputs_np.shape[0])]
                feed_dict[self.t_sequences] = input_np
                feed_dict[self.targets[sample]] = test
                filter_out.extend(self.sess.run(self.feature1, feed_dict))
            filter_out = np.array(filter_out)
            for f in range(num_filters):
                #plot_filter_heat(kernel[:,:,i].T, '../motif/filter%d_head.pdf'%i)
                #filter_possum(kernel[:,:,i].T, 'filter%d'%i, '../motif/possum/filter%d_possum.txt'%(i), False)
                plot_filter_logo(filter_out[:,:,f], filter_size, seqs, '../motif/weblogo/%s/filter%d_logo'%(sample,i),maxpct_t=0.5)

    def test(self):
        print('Testing:')
        ACC_ALL = []
        #AUC_ALL = []
        max_iter = int(self.data_loader.get_test_size() / config.BATCH_SIZE)
        print(max_iter)
        onehots = []
        probbs = [] 
        #label = []
        for idx in range(max_iter):
            feed_dict = {self.t_sequences: np.random.randn(config.BATCH_SIZE,config.TRAIN.TIME_STEPS, config.TRAIN.EMBED_DIM)}
            ACC = []
            #AUC = []
            for id in range(config.TRAIN.CLASSES - 1):
                sample = self.sample_names[id]
                sequences, labels = self.data_loader.get_next_test_batch_by_name(sample)
                feed_dict[self.t_sequences] = sequences
                feed_dict[self.targets[sample]] = labels
                label_onehot= pd.get_dummies(labels).values.tolist()
                #print(label_onehot)
                #exit()
                probs = self.sess.run(self.train_list[sample][2], feed_dict)
                #label.extend(labels)
                onehots.extend(label_onehot)
                probbs.extend(probs)
                #auc = metrics.roc_auc_score(np.array(label_onehot), np.array(probs), average = 'micro')
                pred_label = np.argmax(probs, axis =1)
                acc = np.sum(np.array(pred_label) == np.array(labels)) / len(pred_label)
                
                #AUC.append(auc)
                ACC.append(acc)
            
            ACC_ALL.append(ACC)
            #AUC_ALL.append(AUC)
        print('acc is: ',np.mean(ACC_ALL, axis = 0))
        #print(np.array(onehots).shape)
        auc = metrics.roc_auc_score(np.array(onehots)[:,1], np.array(probbs)[:,1])
        precision, recall, _ = precision_recall_curve(np.array(onehots)[:,1], np.array(probbs)[:,1])
        from matplotlib import pyplot as plt
        plt.step(recall, precision, color='b', alpha=0.2, where='post')
        plt.savefig('test.png')
        ap = average_precision_score(np.array(onehots)[:,1], np.array(probbs)[:,1])
        ps = precision_score(np.array(onehots)[:,1], np.array(probbs)[:,1]>0.5)
        print('pos num: ',sum(np.array(onehots)[:,1]))
        print('total num: ',len(np.array(onehots)[:,1]))
        print('AUC is: ', auc)
        print('AP is: ', ap)
        print('Precision score is: ', ps)
        if 'multi-task' in tl.global_flag['name']:
            s = '_' + config.sample_names[0]
        else:
            s = ''

        namedir = 'predict_result/' + tl.global_flag['name'] + s+ '_%d/'%config.NEG_POS_RATIO
        tl.files.exists_or_mkdir(namedir)
        np.savetxt(namedir + tl.global_flag['name'] + s+ '_%d_'%config.NEG_POS_RATIO  + 'label.txt', np.array(onehots)[:,1],)
        np.savetxt( namedir + tl.global_flag['name'] + s + '_%d_'%config.NEG_POS_RATIO  + 'probs.txt', np.array(probbs)[:,1],)
        #print(np.mean(AUC_ALL, axis = 0))
            
    def save(self, auc):
        if self.pre < auc:
            self.pre = auc
        else:
            return
        # save file into checkpoint dir
        tl.files.save_npz(self.features.all_params, name=self.checkpoint_dir + '/m6aNet_feature_{}_{}.npz'.format(tl.global_flag['name'], auc), sess=self.sess)
        tl.files.save_npz(self.features.all_params, name=self.checkpoint_dir + '/m6aNet_feature_{}.npz'.format(tl.global_flag['name']), sess=self.sess)
        for i in range(len(self.classes_list)):
            sample = self.sample_names[i]
            tl.files.save_npz(self.classes_list[i].all_params, name=self.checkpoint_dir + '/m6aNet_classese_{}_{}_{}.npz'.format(sample, tl.global_flag['name'], auc), sess=self.sess)
            tl.files.save_npz(self.classes_list[i].all_params, name=self.checkpoint_dir + '/m6aNet_classese_{}_{}.npz'.format(sample, tl.global_flag['name']), sess=self.sess)
            
class M6ANetShareAllTarget():

    sample_names = config.sample_names
    def __init__(self, data_loader, model_name):
        self.pre = 0 
        self.data_loader = data_loader
        self.checkpoint_dir = config.CHECK_POINT_DIR

        self.t_sequences = tf.placeholder(tf.float32, [None, config.TRAIN.TIME_STEPS, config.TRAIN.EMBED_DIM])
        # initialize t_targets groups
        self.targets = {}
        for id in range(config.TRAIN.CLASSES - 1):
            sample = self.sample_names[id]
            self.targets[sample] = tf.placeholder(tf.int32, [None,], name = sample + '_target')
        
        FIRST = True
        self.train_list = {}
        self.features = sharedFeatureExtractor(self.t_sequences, 'extractor', reuse = (not FIRST), is_train = True)
        self.classes_list = []
        cost = None
        for id in range(config.TRAIN.CLASSES - 1):
            sample = self.sample_names[id]
            classes = classifier(self.features, sample, reuse = False, is_train = True)
            classes_test = classifier(self.features, sample, reuse = True, is_train = False)
            if cost is None:
                cost = tl.cost.cross_entropy(classes.outputs, self.targets[sample], name = sample + '_entrop')
            else:
                cost = cost + tl.cost.cross_entropy(classes.outputs, self.targets[sample], name = sample + '_entrop')                                   
            #print(classes.all_params)
            #exit()
            #optim_one = tf.train.AdamOptimizer(1e-4, beta1=0.9, name =sample + '_adam_one').minimize(cost, var_list = tl.layers.get_variables_with_name(sample, True, True))
            '''
            if config.TRAIN.OPTIM_CLASSIFIER:
                optim = optim_one
            else:
                optim = optim_all
            '''
            self.train_list[sample] = tf.nn.softmax(classes_test.outputs)
            
            self.classes_list.append(classes)
        self.optim_all = tf.train.AdamOptimizer(1e-4, beta1=0.9, name =sample + '_adam_all').minimize(cost, var_list = tf.trainable_variables())
        self.cost = cost
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
        tl.layers.initialize_global_variables(self.sess)

        self.load()
        #
    def train(self):
        
        for epoch in range(config.TRAIN.MAX_EPOCHS):

            max_iter = int(self.data_loader.get_train_size() / config.BATCH_SIZE)
            print('[INFO] EPOCH', epoch, ' size is ', max_iter)
            feed_dict = {self.t_sequences: np.random.randn(config.BATCH_SIZE * 2,config.TRAIN.TIME_STEPS, config.TRAIN.EMBED_DIM)}

            for id in range(config.TRAIN.CLASSES - 1):
                sample = self.sample_names[id]
                feed_dict[self.targets[sample]] = np.random.randint(0,2,size=config.BATCH_SIZE * 2)

            for idx in range(max_iter):
                # at every iteration, we change samples
                LOSS = 0
                ACC = []
                for id in range(config.TRAIN.CLASSES - 1):
                    for tid in range(config.TRAIN.CLASSES - 1):
                        sample = self.sample_names[tid]
                        feed_dict[self.targets[sample]] = np.zeros(config.BATCH_SIZE * 2)
                    sample = self.sample_names[id]
                    sequences, labels = self.data_loader.get_next_train_batch_by_name(sample)
                    feed_dict[self.t_sequences] = sequences
                    feed_dict[self.targets[sample]] = labels
                    loss,_,probs = self.sess.run([self.cost, self.optim_all, self.train_list[sample],], feed_dict)
                    LOSS += loss
                    pred_label = np.argmax(probs, axis =1)
                    acc = np.sum(np.array(pred_label) == np.array(labels)) / len(pred_label)
                    ACC.append(acc)
                LOSS = LOSS / len(self.sample_names)
                acc = np.mean(ACC)
                mystr = ''
                for id in range(len(ACC)):
                    mystr = mystr + ' %.2f '%(ACC[id])
                    
                if True:#(idx % 100) == 0:
                    percent = (idx + 1) * 50 / max_iter
                    num_arrow = int(percent)
                    num_line = 50 - num_arrow
                    progress_bar = '[' + '>' * num_arrow + '-' * num_line + ']' + '%.2f%% ' % (percent*2) + '  CE: %.3e, acc = %.3f' % (LOSS, acc) + " sep " + mystr+'\r'
                    sys.stdout.write(progress_bar)
                    sys.stdout.flush()
                    
            val_acc = self.evaluate()
            mystr = ''
            for id in range(len(val_acc)):
                mystr = mystr + ' %.2f '%(val_acc[id])

            print('\nEpoch %d/%d'%(epoch, config.TRAIN.MAX_EPOCHS),' accuracy is: %.2f'%np.mean(val_acc), ' sep ' + mystr)
            self.save(np.mean(val_acc))

            if (epoch is not 0) and (epoch % config.TRAIN.IO_POS_EVERY) is 0:
                self.data_loader._diskIO(update_pos = True, update_neg = True)
                #if (epoch is not 0) and (epoch % config.TRAIN.IO_NEG_EVERY) is 0:
                #    self.data_loader._diskIO(update_pos = True, update_neg = True)
    def evaluate(self):
        print('\nEvaluating:')
        ACC_ALL = []
        max_iter = int(self.data_loader.get_valid_size() / config.BATCH_SIZE)
         
        for idx in range(max_iter):
            feed_dict = {self.t_sequences: np.random.randn(config.BATCH_SIZE,config.TRAIN.TIME_STEPS, config.TRAIN.EMBED_DIM)}
            ACC = []
            for id in range(config.TRAIN.CLASSES - 1):
                for tid in range(config.TRAIN.CLASSES - 1):
                    sample = self.sample_names[tid]
                    feed_dict[self.targets[sample]] = np.zeros(config.BATCH_SIZE * 2)
                sample = self.sample_names[id]
                sequences, labels = self.data_loader.get_next_valid_batch_by_name(sample)
                feed_dict[self.t_sequences] = sequences
                feed_dict[self.targets[sample]] = labels
                    
                probs = self.sess.run(self.train_list[sample], feed_dict)
                    
                pred_label = np.argmax(probs, axis =1)
                acc = np.sum(np.array(pred_label) == np.array(labels)) / len(pred_label)
                #print('prediction: ', pred_label)
                #print('target: ', labels)
                ACC.append(acc)
            #print('\nAll samples acc is:',ACC)
            #print('mean acc is: ', np.mean(ACC))
            ACC_ALL.append(ACC)
        #print('\nMean ACC is : ', np.mean(ACC_ALL))
        return np.mean(ACC_ALL, axis = 0)

    def load(self):
        '''
        directly load from chekpoint dir
        '''
        name = self.checkpoint_dir + '/m6aNet_feature_{}.npz'.format(tl.global_flag['name'])
        if os.path.exists(name):
            tl.files.load_and_assign_npz(sess = self.sess, network = self.features, name = name)
        for i in range(len(self.classes_list)):
            sample = self.sample_names[i]
            name = self.checkpoint_dir + '/m6aNet_classese_{}_{}.npz'.format(sample, tl.global_flag['name'])
            if os.path.exists(name):
                tl.files.load_and_assign_npz(network = self.classes_list[i], name=name, sess=self.sess)

    def save(self, auc):
        if self.pre < auc:
            self.pre = auc
        else:
            return
        # save file into checkpoint dir
        tl.files.save_npz(self.features.all_params, name=self.checkpoint_dir + '/m6aNet_feature_{}.npz'.format(tl.global_flag['name']), sess=self.sess)
        for i in range(len(self.classes_list)):
            sample = self.sample_names[i]
            tl.files.save_npz(self.classes_list[i].all_params, name=self.checkpoint_dir + '/m6aNet_classese_{}_{}.npz'.format(sample, tl.global_flag['name']), sess=self.sess)

class AllTogether():
    sample_names = config.sample_names
    def __init__(self, data_loader, model1 = 'sharedFeatureExtractor', model2 = 'classifier'):
        self.pre = 0 
        self.data_loader = data_loader
        self.checkpoint_dir = config.CHECK_POINT_DIR

        self.t_sequences = tf.placeholder(tf.float32, [None, config.TRAIN.TIME_STEPS, config.TRAIN.EMBED_DIM])
        # initialize t_targets groups
        
        self.targets = tf.placeholder(tf.int32, [None,], name = 'target')

        self.features, self.feature1 =  sharedFeatureExtractor2(self.t_sequences, 'extractor', is_train = True)#eval(model1 + "(self.t_sequences, 'extractor', is_train = True)")
        self.classes = classifier(self.features, 'together', reuse = False, is_train = True)
        self.cost = tl.cost.cross_entropy(self.classes.outputs, self.targets, name ='entrop')
        self.optim = tf.train.AdamOptimizer(1e-4, beta1=0.9, name ='adam_all').minimize(self.cost, var_list = self.classes.all_params)
        self.probs = tf.nn.softmax(self.classes.outputs)
        
        gpu_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        gpu_config.gpu_options.allow_growth = True
        
        self.sess = tf.Session(config=gpu_config)#tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
        tl.layers.initialize_global_variables(self.sess)

        self.load()
        #
    def train(self):
        
        for epoch in range(config.TRAIN.MAX_EPOCHS):

            max_iter = int(self.data_loader.get_train_size() / config.BATCH_SIZE)
            print('[INFO] EPOCH', epoch, ' size is ', max_iter)
            feed_dict = {self.t_sequences: np.random.randn(config.BATCH_SIZE * 2,config.TRAIN.TIME_STEPS, config.TRAIN.EMBED_DIM)}

            for idx in range(max_iter):
                # at every iteration, we change samples
                LOSS = 0
                ACC = []
                AUC = []
                for id in range(config.TRAIN.CLASSES - 1):
                    sample = self.sample_names[id]
                    sequences, labels = self.data_loader.get_next_train_batch_by_name(sample)
                    feed_dict[self.t_sequences] = sequences
                    feed_dict[self.targets] = labels
                    label_onehot= pd.get_dummies(labels).values.tolist()
                    loss,_,probs = self.sess.run([self.cost, self.optim, self.probs], feed_dict)
                    auc = metrics.roc_auc_score(np.array(label_onehot), np.array(probs), average = 'micro')
                    AUC.append(auc)
                    
                    LOSS += loss
                    pred_label = np.argmax(probs, axis =1)
                    acc = np.sum(np.array(pred_label) == np.array(labels)) / len(pred_label)
                    ACC.append(acc)
                LOSS = LOSS / len(self.sample_names)
                acc = np.mean(ACC)
                mystr = ''
                aucstr = ''
                for id in range(len(ACC)):
                    mystr = mystr + ' %.2f '%(ACC[id])
                    aucstr = aucstr + ' %.2f '%(AUC[id])
                if True:#(idx % 100) == 0:
                    percent = (idx + 1) * 50 / max_iter
                    num_arrow = int(percent)
                    num_line = 50 - num_arrow
                    progress_bar = '[' + '>' * num_arrow + '-' * num_line + ']' + '%.2f%% ' % (percent*2) + '  CE: %.3e, acc = %.3f' % (LOSS, acc) + " sep " + mystr+ ' auc ' + aucstr + '\r'
                    sys.stdout.write(progress_bar)
                    sys.stdout.flush()
                    
            val_acc, val_auc = self.evaluate()
            mystr = ''
            aucstr = ''
            for id in range(len(val_acc)):
                mystr = mystr + ' %.2f '%(val_acc[id])
                aucstr = aucstr + ' %.2f '%(val_auc[id])
                
            print('\nEpoch %d/%d'%(epoch, config.TRAIN.MAX_EPOCHS),' accuracy is: %.3f'%np.mean(val_acc), ' sep ' + mystr + ' auc ' + aucstr)
            self.save(np.mean(val_acc))

            if (epoch % config.TRAIN.IO_POS_EVERY) is 0:
                self.data_loader._diskIO(update_pos = True, update_neg = True)
                #if (epoch is not 0) and (epoch % config.TRAIN.IO_NEG_EVERY) is 0:
                #    self.data_loader._diskIO(update_pos = True, update_neg = True)
    def evaluate(self):
        print('\nEvaluating:')
        ACC_ALL = []
        AUC_ALL = []
        max_iter = int(self.data_loader.get_valid_size() / config.BATCH_SIZE)
         
        for idx in range(max_iter):
            feed_dict = {self.t_sequences: np.random.randn(config.BATCH_SIZE,config.TRAIN.TIME_STEPS, config.TRAIN.EMBED_DIM)}
            ACC = []
            AUC = []
            for id in range(config.TRAIN.CLASSES - 1):
                sample = self.sample_names[id]
                sequences, labels = self.data_loader.get_next_valid_batch_by_name(sample)
                feed_dict[self.t_sequences] = sequences
                feed_dict[self.targets] = labels
                label_onehot= pd.get_dummies(labels).values.tolist()
                probs = self.sess.run(self.probs, feed_dict)
                auc = metrics.roc_auc_score(np.array(label_onehot), np.array(probs), average = 'micro')
                pred_label = np.argmax(probs, axis =1)
                acc = np.sum(np.array(pred_label) == np.array(labels)) / len(pred_label)

                AUC.append(auc)
                ACC.append(acc)

            ACC_ALL.append(ACC)
            AUC_ALL.append(AUC)

        return np.mean(ACC_ALL, axis = 0), np.mean(AUC_ALL, axis = 0)

    def load(self):
        '''
        directly load from chekpoint dir
        '''
        import os
        name = self.checkpoint_dir + '/m6aNet_alltogether_{}.npz'.format(tl.global_flag['name'])
        print(name)
        if os.path.exists(name):
            tl.files.load_and_assign_npz(sess = self.sess, network = self.classes, name = name)


    def test_old(self):
        print('\nTesting:')
        ACC_ALL = []
        AUC_ALL = []
        
        for id in range(config.TRAIN.CLASSES - 1): 
            sample = self.sample_names[id]
            max_iter = math.ceil(self.data_loader.get_test_size_by_name(sample) / (config.BATCH_SIZE)*3)
            feed_dict = {self.t_sequences: np.random.randn(config.BATCH_SIZE,config.TRAIN.TIME_STEPS, config.TRAIN.EMBED_DIM)}
            
            probs_list = []
            labels_list = []
            
            for idx in range(max_iter):

                sequences, labels = self.data_loader.get_next_test_batch_by_name(sample)
                feed_dict[self.t_sequences] = sequences
                feed_dict[self.targets[sample]] = labels
                label_onehot= pd.get_dummies(labels).values.tolist()
                probs = self.sess.run(self.train_list[sample][2], feed_dict)
                #auc = metrics.roc_auc_score(np.array(label_onehot), np.array(probs), average = 'micro')
                #pred_label = np.argmax(probs, axis =1)
                #acc = np.sum(np.array(pred_label) == np.array(labels)) / len(pred_label)
                labels_list.extend(label_onehot)
                probs_list.extend(probs)
            print('Doing computing')
            # compute accuracy, precision_score, auc
            #label_onehot= pd.get_dummies(labels_list).values.tolist()
            prob = np.argmax(probs_list, axis = 1)
            print('Calculating AUC')
            auc = metrics.roc_auc_score(prob,labels_list, average = 'micro')
            print('Calculating F1 score')
            f1 = f1_score(prob,labels_list)
            print('Sample {} f1: {} auc: {}'.format(sample,f1, auc))

    def test(self):
        print('Testing:')
        ACC_ALL = []
        #AUC_ALL = []
        max_iter = int(self.data_loader.get_test_size() / config.BATCH_SIZE)
        print(max_iter)
        onehots = []
        probbs = [] 
        for idx in range(max_iter):
            feed_dict = {self.t_sequences: np.random.randn(config.BATCH_SIZE,config.TRAIN.TIME_STEPS, config.TRAIN.EMBED_DIM)}
            ACC = []
            #AUC = []
            for id in range(config.TRAIN.CLASSES - 1):
                sample = self.sample_names[id]
                sequences, labels = self.data_loader.get_next_test_batch_by_name(sample)
                feed_dict[self.t_sequences] = sequences
                feed_dict[self.targets] = labels
                label_onehot= pd.get_dummies(labels).values.tolist()
                #print(label_onehot)
                #exit()
                probs = self.sess.run(self.probs, feed_dict)
                onehots.extend(label_onehot)
                probbs.extend(probs)
                #auc = metrics.roc_auc_score(np.array(label_onehot), np.array(probs), average = 'micro')
                pred_label = np.argmax(probs, axis =1)
                acc = np.sum(np.array(pred_label) == np.array(labels)) / len(pred_label)
                
                #AUC.append(auc)
                ACC.append(acc)
            
            ACC_ALL.append(ACC)
            #AUC_ALL.append(AUC)
        print(np.mean(ACC_ALL, axis = 0))
        auc = metrics.roc_auc_score(np.array(onehots), np.array(probbs), average = 'micro')
        print('AUC is: ', auc)
        ap = average_precision_score(np.array(onehots)[:,1], np.array(probbs)[:,1])
        print('AP is: ', ap)
        #print(np.mean(AUC_ALL, axis = 0))
            
    def save(self, auc):
        if self.pre < auc:
            self.pre = auc
        else:
            return
        # save file into checkpoint dir
        tl.files.save_npz(self.classes.all_params, name=self.checkpoint_dir + '/m6aNet_alltogether_{}_{}.npz'.format(tl.global_flag['name'], auc), sess=self.sess)
        tl.files.save_npz(self.classes.all_params, name=self.checkpoint_dir + '/m6aNet_alltogether_{}.npz'.format(tl.global_flag['name']), sess=self.sess)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='../data/seq_samples/', help='training data directory')
    parser.add_argument('--name', type=str, default='srmd', help='srmd_CHANNEL_STACK, anything you want')
    parser.add_argument('--mode', type=str, default='train', help='train or evaluate')
    #parser.add_argument('--model_name', type=str, default='Selector', help='train or evaluate')
    parser.add_argument('--gpu', type=str, default="0", help='GPU ID')
    parser.add_argument('--fold', type=str, default="0", help='fold num')
    parser.add_argument('--data', type=str, default="full_data", help='fold num')
    parser.add_argument('--sn', type=int, default=2, help='species num')
    args = parser.parse_args()
    import os

    config.TRAIN.CLASSES = args.sn
    if 'deepm6aseq' in args.name:
        config.TRAIN.DROPOUT_KEEP = 0.5
    else:
        config.TRAIN.DROPOUT_KEEP = 0.5
    if args.data == 'full_data':
        pos_dir = '/home/xiongyuanpeng/Research/m6a_prediction/data/sequence_samples/positive_samples'
        neg_dir = '/home/xiongyuanpeng/Research/m6a_prediction/data/sequence_samples/negative_samples'
        config.train_suffix = '_1000_100'#'_100_sramp_down'#
        config.val_suffix ='_1000_100_self_processed'# '_100_sramp_down'#
        config.test_suffix = config.val_suffix#'_100_sramp'
        config.NEG_IO_STEP = 0.03
    else:
        print(args.data)
        if 'sramp' in args.data:
            pos_dir = '/home/xiongyuanpeng/Research/m6a_prediction/data/sramp/positive_samples'
            neg_dir = '/home/xiongyuanpeng/Research/m6a_prediction/data/sramp/negative_samples'
            config.train_suffix = '_100_sramp_down'#'_1000_100'#
            config.val_suffix ='_100_sramp_down'#'_1000_100_self_processed'# 
            config.test_suffix = '_100_sramp'
            config.NEG_IO_STEP = 0.1
        else:
            if 'deepm6aseq' in args.data:
                pos_dir = '/home/xiongyuanpeng/DeepM6ASeq/data/positive_samples'
                neg_dir = '/home/xiongyuanpeng/DeepM6ASeq/data/negative_samples'
                config.train_suffix = '_1000_100'#'_1000_100'#
                config.val_suffix ='_1000_100_down'#'_1000_100_self_processed'# 
                config.test_suffix = '_1000'
                config.NEG_IO_STEP = 1
                config.POS_IO_STEP = 1


    tl.global_flag['input_dir'] = args.input_dir

    if config.TRAIN.CLASSES == 2:
        config.sample_names[0] = config.sample_names[int(args.fold)]
        sample = '_' +config.sample_names[0]
    else:
        sample = '_multi-task_%d'%(config.TRAIN.CLASSES - 1)

    model_name = config.model_setting + '_' +args.name +  '_' + args.data + sample #+ '_fc1'#config.sample_names[0] #+'_fc1'
    tl.global_flag['name'] = model_name
    tl.files.exists_or_mkdir(config.CHECK_POINT_DIR)
    os.environ['CUDA_VISIBLE_DEVICES']= args.gpu

    if args.mode == 'test':
        config.NEG_POS_RATIO = 10
        if config.TRAIN.CLASSES is not 2:
            # should tansfer it to 2
            config.TRAIN.CLASSES = 2
            # let fold to indicate species
            config.sample_names[0] = config.sample_names[int(args.fold)]
    if args.mode == 'train':
        config.NEG_POS_RATIO = 1
        load_train = True
        from datetime import datetime 
        date = str(datetime.now()).split(' ')[0]
        tl.files.exists_or_mkdir('./code_bkp/%s'%date)
        os.system("zip ./code_bkp/%s/%s.zip *.py"%(date,config.model_setting + '_' + args.name))
    else:

        load_train = False
    if args.mode == 'motif':
        data_loader = None
    else:
        data_loader = SequencesProvider(pos_dir= pos_dir,neg_dir= neg_dir, load_train = load_train)
        #data_loader = SequencesProvider(pos_dir= '/home/xiongyuanpeng/DeepM6ASeq/data/positive_samples',neg_dir= /home/xiongyuanpeng/DeepM6ASeq/data/negative_samples', load_train = load_train)
    #model_type = args.name
    if args.name == 'deepm6aseq':
        model1 = 'DeepM6ASeq_pre'
        model2 =  'DeepM6ASeq'
    else:
        if 'mass' in args.name:# == 'mass':
            model1 = 'sharedFeatureExtractor2'
        else:
            model1 = 'sharedFeatureExtractor_nodropout'
        if CLASS_SEQ:
            model2 = 'classifierSequences'
        else:
            model2 = 'classifier'
    m6anet = M6ANetShare(data_loader, model1 = model1, model2 = model2)
    if args.mode == 'train':
        m6anet.train()
    elif args.mode == 'test':
        m6anet.test()
    elif args.mode =='evaluate':
        m6anet.evaluate()
    elif args.mode == 'motif':
        m6anet.motif()
