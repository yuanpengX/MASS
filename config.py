# encoding: utf-8
# author: xiongyuanpeng
# 2018-11-14
from easydict import EasyDict as edict

config = edict()

config.TRAIN = edict() 
config.TRAIN.EMBED_DIM = 4 # input lenght
config.TRAIN.RNN_HIDDEN = 64 # hidden dim for rnn state
config.TRAIN.KERNEL = '18_18_18'
config.TRAIN.TIME_STEPS = 101 # sequence lengt
config.TRAIN.RNN_STEPS = 100#125 # sequence length
config.TRAIN.FC = 128 # hidden layer dim for full connected
config.TRAIN.CLASSES ='_fixed'# output categorie
config.TRAIN.STACK_DEPTH =  5 # stacking bypass structre
config.TRAIN.INIT_LR = 1e-2
config.TRAIN.beta1 = 0.9
config.TRAIN.MAX_EPOCHS = 50
config.TRAIN.SAVE_EVERY = 5 # save every 5 epochs
config.TRAIN.IO_POS_EVERY = 1
config.TRAIN.IO_NEG_EVERY = 2
config.TRAIN.LAMBDA = 0
config.TRAIN.OPTIM_CLASSIFIER = False  # only optimize classifier?
config.TRAIN.DROPOUT = True
config.TRAIN.DROPOUT_KEEP =0.5
config.CHECK_POINT_DIR  = '../checkpoint/'
#config.SAMPLE = int(26/config.TRAIN.CLASSES)
config.POS_IO_STEP = 1
config.NEG_IO_STEP = 0.1
config.DEBUG = False
config.BATCH_SIZE  = 350
config.DEBUG_SAMPLE_SIZE = 200
config.NEG_POS_RATIO=1
config.DO_MULTI = False
config.sample_names =  ['hg19','panTro4','rheMac8','rn5','mm10','susScr3','danRer10','susScr3','rn5','rheMac8','panTro4','susScr3','mm10', 'rheMac8','panTro4','hg19','rn5','mm10','rheMac8','panTro4','rn5', 'susScr3','danRer10', 'TAIR10','sacCer3', 'BDGP6', 'ASM584v2', 'ASM676v1',]

config.model_setting = 'rnn{}_kerel{}_fc{}_lr{}_classnum{}_drop{}'.format(config.TRAIN.RNN_HIDDEN, config.TRAIN.KERNEL, config.TRAIN.FC, config.TRAIN.INIT_LR, config.TRAIN.CLASSES, config.TRAIN.DROPOUT_KEEP)               
config.group_setting = {'G1':['hg19','panTro4','rheMac8',], 'G2':['mm10', 'rn5',], 'G3':['susScr3','danRer10',]}
#config.TRAIN.DROPOUT_KEEP = 1
