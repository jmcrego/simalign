# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import io
import os
import sys
from shutil import copyfile
from dataset import Vocab, Embeddings

class Config():

    def __init__(self, argv):
        self.usage="""usage: {}
*  -mdir          FILE : directory to save/restore models

   -seq_size       INT : sentences larger than this number of src/tgt words are filtered out [50]
   -batch_size     INT : number of examples per batch [32]
   -seed           INT : seed for randomness [1234]
   -debug              : debug mode 
   -h                  : this message

 [LEARNING OPTIONS]
*  -trn           FILE : training data
   -dev           FILE : validation data

   -src_voc       FILE : vocabulary of src words (needed to initialize learning)
   -tgt_voc       FILE : vocabulary of tgt words (needed to initialize learning)
   -src_emb       FILE : embeddings of src words (needed to initialize learning)
   -tgt_emb       FILE : embeddings of tgt words (needed to initialize learning)
   -src_emb_size   INT : size of src embeddings if -src_emb not used
   -tgt_emb_size   INT : size of tgt embeddings if -tgt_emb not used
   
   -src_lstm_size  INT : hidden units for src bi-lstm [256]
   -tgt_lstm_size  INT : hidden units for tgt bi-lstm [256]

   -dropout      FLOAT : dropout ratio [0.3]
   -error       STRING : error function (mse, exp, lse) [mse]
   -sloss        FLOAT : loss = wloss + sloss*serror [0.0]
   -r            FLOAT : r used when error=='lse' [1.0]
   -aggr        STRING : aggregation function (sum, lse) [sum] NOT USED
   -lr_method   STRING : GD method either: adam, adagrad, adadelta, sgd, rmsprop [adam]
   -lr           FLOAT : initial learning rate [0.001]
   -lr_decay     FLOAT : learning rate decay if lr_method is sgd [0.9]
   -clip         FLOAT : gradient clipping value (0.0 for no clipping) [0.0]

   -p_unpair     FLOAT : probability of unpaired sentences [0.0]
   -p_delete     FLOAT : probability of sentences with deleted sequences of words [0.0]
   -p_extend     FLOAT : probability of sentences with extended sequences of words [0.0]
   -p_replace    FLOAT : probability of sentences with replaced sequences of words [0.0]

   -max_sents      INT : Consider this number of sentences per batch (0 for all) [0]
   -n_epochs       INT : train for this number of epochs [1]
   -report_every   INT : report every this many batches [100]

 [INFERENCE OPTIONS]
*  -epoch          INT : epoch to use ([mdir]/epoch[epoch] must exist)
*  -tst           FILE : testing data
   -sim            SIM : one of these: last, mean, max [last]
   -show_matrix        : output formatted alignment matrix
   -show_align         : output source/target alignment matrix
   -show_sim           : output source/target similarity vectors
   -show_svg           : output alignment matrix using svg-like html format
   -mark_unks          : mark OOVs 

+ Options marked with * must be set. The rest have default values.
+ If -mdir exists in learning mode, learning continues after restoring the last model
+ Training data is shuffled at every epoch
+ -show_sim, -show_align can be used at the same time
""".format(sys.argv.pop(0))

        self.src_voc = None
        self.tgt_voc = None
        self.src_emb = None
        self.tgt_emb = None
        self.src_voc_size = None
        self.tgt_voc_size = None
        self.src_emb_size = None
        self.tgt_emb_size = None
        self.mdir = None
        self.epoch = None
        self.trn = None
        self.dev = None
        self.tst = None
        self.emb_src = None # object with embeddings
        self.emb_tgt = None # object with embeddings

        self.share = False

        self.src_lstm_size = 256
        self.tgt_lstm_size = 256

        self.p_unpair = 0.0
        self.p_swap = 0.0
        self.p_delete = 0.0
        self.p_extend = 0.0
        self.p_replace = 0.0

        self.dropout = 0.3
        self.lr = 0.001
        self.lr_decay = 0.9
        self.lr_method = "adam"
        self.clip = 0.0

        self.seq_size = 50
        self.batch_size = 32
        self.max_sents = 0
        self.n_epochs = 1
        self.last_epoch = 0 # epochs already run
        self.seed = 1234
        self.report_every = 100
        self.debug = False
        self.mark_unks = False
        self.r = 1.0
        self.sloss = 0.0
        self.error = "mse"
        self.aggr = "sum"
        self.sim = "last"

        self.show_matrix = False
        self.show_svg = False
        self.show_sim = False
        self.show_align = False

        self.parse(sys.argv)

        tf.set_random_seed(self.seed)
        np.random.seed(self.seed)

        if not self.mdir:
            sys.stderr.write("error: Missing -mdir option\n{}".format(self.usage))
            sys.exit()

        if self.tst: self.inference()
        if self.trn: self.learn()
        return

    def inference(self):
        self.dropout = 0.0
        self.seq_size = 0
        if not self.epoch:
            sys.stderr.write("error: Missing -epoch option\n{}".format(self.usage))
            sys.exit()
        if not os.path.exists(self.tst):
            sys.stderr.write('error: -tst file {} cannot be find\n'.format(self.tst))
            sys.exit()
        if not os.path.exists(self.mdir + '/epoch' + self.epoch + '.index'):
            sys.stderr.write('error: -epoch file {} cannot be find\n'.format(self.mdir + '/epoch' + self.epoch + '.index'))
            sys.exit()
        if not os.path.exists(self.mdir + '/topology'): 
            sys.stderr.write('error: topology file: {} cannot be find\n'.format(self.mdir + '/topology'))
            sys.exit()
        if not os.path.exists(self.mdir + '/vocab_src'): 
            sys.stderr.write('error: vocab_src file: {} cannot be find\n'.format(self.mdir + '/vocab_src'))
            sys.exit()
        if not os.path.exists(self.mdir + '/vocab_tgt'): 
            sys.stderr.write('error: vocab_tgt file: {} cannot be find\n'.format(self.mdir + '/vocab_tgt'))
            sys.exit()
        argv = []
        with open(self.mdir + "/topology", 'r') as f:
            for line in f:
                opt, val = line.split()
                argv.append('-'+opt)
                argv.append(val)
        self.parse(argv) ### this overrides options passed in command line

        ### read vocabularies
        self.voc_src = Vocab(self.mdir + "/vocab_src") 
        self.voc_tgt = Vocab(self.mdir + "/vocab_tgt")
        return  

    def learn(self):
        if not os.path.exists(self.trn):
            sys.stderr.write('error: -trn file {} cannot be find\n'.format(self.trn))
            sys.exit()
        if self.dev is not None and not os.path.exists(self.dev):
            sys.stderr.write('error: -dev file {} cannot be find\n'.format(self.dev))
            sys.exit()
        ###
        ### continuation
        ###
        if os.path.exists(self.mdir): 
            if not os.path.exists(self.mdir + '/topology'): 
                sys.stderr.write('error: topology file: {} cannot be find\n'.format(self.mdir + '/topology'))
                sys.exit()
            if not os.path.exists(self.mdir + '/vocab_src'): 
                sys.stderr.write('error: vocab_src file: {} cannot be find\n'.format(self.mdir + '/vocab_src'))
                sys.exit()
            if not os.path.exists(self.mdir + '/vocab_tgt'): 
                sys.stderr.write('error: vocab_tgt file: {} cannot be find\n'.format(self.mdir + '/vocab_tgt'))
                sys.exit()
            if not os.path.exists(self.mdir + '/checkpoint'): 
                sys.stderr.write('error: checkpoint file: {} cannot be find\ndelete dir {} ???\n'.format(self.mdir + '/checkpoint', self.mdir))
                sys.exit()

            argv = []
            with open(self.mdir + "/topology", 'r') as f:
                for line in f:
                    opt, val = line.split()
                    argv.append('-'+opt)
                    argv.append(val)
            self.parse(argv) ### this overrides options passed in command line
            ### read vocabularies
            self.voc_src = Vocab(self.mdir + "/vocab_src") 
            self.voc_tgt = Vocab(self.mdir + "/vocab_tgt")
            ### update last epoch
            for e in range(999,0,-1):
                if os.path.exists(self.mdir+"/epoch{}.index".format(e)): 
                    self.last_epoch = e
                    break
            print("learning continuation: last epoch is {}".format(self.last_epoch))
        ###
        ### learning from scratch
        ###
        else:
            self.voc_src = Vocab(self.src_voc) ### read file or config/vocab_src if file is not set
            self.voc_tgt = Vocab(self.tgt_voc) ### read file or config/vocab_tgt if file is not set
            self.src_voc_size = self.voc_src.length
            self.tgt_voc_size = self.voc_tgt.length

            if not os.path.exists(self.mdir): os.makedirs(self.mdir)
            #copy vocabularies
            copyfile(self.src_voc, self.mdir + "/vocab_src")
            copyfile(self.tgt_voc, self.mdir + "/vocab_tgt")
            #read embeddings
            self.emb_src = Embeddings(self.src_emb,self.voc_src,self.src_emb_size) ### read file or use emb_src.length if file is not set
            self.src_emb_size = self.emb_src.dim
            self.emb_tgt = Embeddings(self.tgt_emb,self.voc_tgt,self.tgt_emb_size) ### read file or use emb_tgt.length if file is not set
            self.tgt_emb_size = self.emb_tgt.dim
            #write topology file
            with open(self.mdir + "/topology", 'w') as f: 
                for opt, val in vars(self).items():
                    if opt.startswith("src") or opt.startswith("tgt"): f.write("{} {}\n".format(opt,val))
            print("learning from scratch")
        return  


    def parse(self, argv):
        while len(argv):
            tok = argv.pop(0)
            if (tok=="-mdir" and len(argv)):
                self.mdir = argv.pop(0)
            elif (tok=="-epoch" and len(argv)):
                self.epoch = argv.pop(0)
            elif (tok=="-src_voc" and len(argv)):
                self.src_voc = argv.pop(0)
            elif (tok=="-tgt_voc" and len(argv)):
                self.tgt_voc = argv.pop(0)
            elif (tok=="-src_emb" and len(argv)):
                self.src_emb = argv.pop(0)
            elif (tok=="-tgt_emb" and len(argv)):
                self.tgt_emb = argv.pop(0)
            elif (tok=="-src_voc_size" and len(argv)):
                self.src_voc_size = int(argv.pop(0))
            elif (tok=="-tgt_voc_size" and len(argv)):
                self.tgt_voc_size = int(argv.pop(0))
            elif (tok=="-src_emb_size" and len(argv)):
                self.src_emb_size = int(argv.pop(0))
            elif (tok=="-tgt_emb_size" and len(argv)):
                self.tgt_emb_size = int(argv.pop(0))
            elif (tok=="-trn" and len(argv)):
                self.trn = argv.pop(0)
            elif (tok=="-dev" and len(argv)):
                self.dev = argv.pop(0)
            elif (tok=="-tst" and len(argv)):
                self.tst = argv.pop(0)
            elif (tok=="-max_sents" and len(argv)):
                self.max_sents = int(argv.pop(0))
            elif (tok=="-debug"):
                self.debug = True
            elif (tok=="-seed" and len(argv)):
                self.seed = int(argv.pop(0))
            elif (tok=="-report_every" and len(argv)):
                self.report_every = int(argv.pop(0))
            elif (tok=="-n_epochs" and len(argv)):
                self.n_epochs = int(argv.pop(0))

            elif (tok=="-src_lstm_size" and len(argv)):
                self.src_lstm_size = int(argv.pop(0))
            elif (tok=="-tgt_lstm_size" and len(argv)):
                self.tgt_lstm_size = int(argv.pop(0))

            elif (tok=="-p_unpair" and len(argv)):
                self.p_unpair = float(argv.pop(0))
            elif (tok=="-p_swap" and len(argv)):
                self.p_swap = float(argv.pop(0))
            elif (tok=="-p_delete" and len(argv)):
                self.p_delete = float(argv.pop(0))
            elif (tok=="-p_extend" and len(argv)):
                self.p_extend = float(argv.pop(0))
            elif (tok=="-p_replace" and len(argv)):
                self.p_replace = float(argv.pop(0))

            elif (tok=="-seq_size" and len(argv)):
                self.seq_size = int(argv.pop(0))
            elif (tok=="-batch_size" and len(argv)):
                self.batch_size = int(argv.pop(0))
            elif (tok=="-dropout" and len(argv)):
                self.dropout = float(argv.pop(0))
            elif (tok=="-lr" and len(argv)):
                self.lr = float(argv.pop(0))
            elif (tok=="-lr_decay" and len(argv)):
                self.lr_decay = float(argv.pop(0))
            elif (tok=="-lr_method" and len(argv)):
                self.lr_method = argv.pop(0)
            elif (tok=="-clip" and len(argv)):
                self.clip = float(argv.pop(0))
            elif (tok=="-r" and len(argv)):
                self.r = float(argv.pop(0))
            elif (tok=="-error" and len(argv)):
                self.error = argv.pop(0)
            elif (tok=="-aggr" and len(argv)):
                self.aggr = argv.pop(0)
            elif (tok=="-sloss" and len(argv)):
                self.sloss = float(argv.pop(0))

            elif (tok=="-sim" and len(argv)):
                self.sim = argv.pop(0)

            elif (tok=="-show_matrix"):
                self.show_matrix = True
            elif (tok=="-show_svg"):
                self.show_svg = True
            elif (tok=="-show_sim"):
                self.show_sim = True
            elif (tok=="-show_align"):
                self.show_align = True
            elif (tok=="-mark_unks"):
                self.mark_unks = True

            elif (tok=="-h"):
                sys.stderr.write("{}".format(self.usage))
                sys.exit()

            else:
                sys.stderr.write('error: unparsed {} option\n'.format(tok))
                sys.stderr.write("{}".format(self.usage))
                sys.exit()

    def write_config(self):
        if not os.path.exists(self.mdir): 
            os.makedirs(self.mdir)
        file = self.mdir + "/epoch"+str(self.last_epoch)+".config"
        with open(file,"w") as f:
            for name, val in vars(self).items():
                if name=="usage" or name.startswith("emb_") or name.startswith("voc_"): continue
                f.write("{} {}\n".format(name,val))



