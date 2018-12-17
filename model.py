# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import math
import sys
import os
import time
from random import randint
from config import Config
from dataset import minibatches
from visualize import Visualize

class Score():
    def __init__(self):
        self.TP = 0
        self.TN = 0
        self.FP = 0
        self.FN = 0
        self.A = 0.0
        self.P = 0.0
        self.R = 0.0
        self.F = 0.0

    def add_batch(self, p, r): ### prediction, reference
        #reference contains:
        # +1: aligned
        # -1: unaligned
        #  0: padded
        p_times_r = p * r
        TRUE = np.greater(p_times_r, np.zeros_like(r)) ### matix with true predictions
        FALS = np.less(p_times_r, np.zeros_like(r)) ### matix with false predictions
        POS = np.greater(p, np.zeros_like(r)) ### matix with positive predictions (aligned words)
        NEG = np.less(p, np.zeros_like(r)) ### matix with negative predictions (unaligned wods)
        ### Attention: predictions p==0.000 are not considered 
        self.TP += np.count_nonzero(np.logical_and(TRUE, POS))
        self.TN += np.count_nonzero(np.logical_and(TRUE, NEG))
        self.FP += np.count_nonzero(np.logical_and(FALS, POS))
        self.FN += np.count_nonzero(np.logical_and(FALS, NEG))

    def summarize(self):
        self.A, self.P, self.R, self.F = 0.0, 0.0, 0.0, 0.0
        if self.TP + self.FP > 0: self.P = 1. * self.TP / (self.TP + self.FP) #true positives out of all that were predicted positive
        if self.TP + self.FN > 0: self.R = 1. * self.TP / (self.TP + self.FN) #true positives out of all that were actually positive
        if self.P + self.R > 0.0: self.F = 2. * self.P * self.R / (self.P + self.R) #F-measure
        if self.TP + self.TN + self.FP + self.FN > 0: self.A = 1.0 * (self.TP + self.TN) / (self.TP + self.TN + self.FP + self.FN) #Accuracy
        self.results = "A{:.4f},P{:.4f},R{:.4f},F{:.4f} | TP:{},TN:{},FP:{},FN:{}".format(self.A,self.P,self.R,self.F,self.TP,self.TN,self.FP,self.FN)

class Model():
    def __init__(self, config):
        self.config = config
        self.sess = None

    def embedding_initialize(self,NS,ES,embeddings):
        if embeddings is not None: 
            m = embeddings.matrix
        else:
            sys.stderr.write("embeddings randomly initialized\n")
            m = tf.random_uniform([NS, ES], minval=-0.1, maxval=0.1)
        return m

###################
### build graph ###
###################

    def add_placeholders(self):
        self.input_src  = tf.placeholder(tf.int32, shape=[None,None],        name="input_src")  # Shape: batch_size x |Fj|  (all sentences Fj are equally sized (padded if needed))  
        self.input_tgt  = tf.placeholder(tf.int32, shape=[None,None],        name="input_tgt")  # Shape: batch_size x |Ei|  (all sentences Ej are equally sized (padded if needed))  
        self.input_ali  = tf.placeholder(tf.float32, shape=[None,None,None], name="input_ali")
        self.len_src    = tf.placeholder(tf.int32, shape=[None], name="len_src")
        self.len_tgt    = tf.placeholder(tf.int32, shape=[None], name="len_tgt")
        self.lr         = tf.placeholder(tf.float32, shape=[], name="lr")

#        n_pairs = 1.0 + tf.count_nonzero(self.input_ali, dtype=tf.float32)
#        n_ones = 1.0 + tf.count_nonzero(tf.equal(self.input_ali,tf.zeros(tf.shape(self.input_ali))+1.0), dtype=tf.float32)
#        self.scale = n_pairs / n_ones

    def add_model(self):
        BS = tf.shape(self.input_src)[0] #batch size
        KEEP = 1.0-self.config.dropout   # keep probability for embeddings dropout Ex: 0.7
#        print("KEEP={}".format(KEEP))

        ###
        ### src-side
        ###
        NW = self.config.src_voc_size #src vocab
        ES = self.config.src_emb_size #src embedding size
        L1 = self.config.src_lstm_size #src lstm size
        #print("SRC NW={} ES={}".format(NW,ES))
        with tf.device('/cpu:0'), tf.variable_scope("embedding_src"):
            self.LT_src = tf.get_variable(initializer = self.embedding_initialize(NW, ES, self.config.emb_src), dtype=tf.float32, name="embeddings_src")
            self.embed_src = tf.nn.embedding_lookup(self.LT_src, self.input_src, name="embed_src")
            self.embed_src = tf.nn.dropout(self.embed_src, keep_prob=KEEP)

        with tf.variable_scope("lstm_src"):
            #print("SRC L1={}".format(L1))
            cell_fw = tf.contrib.rnn.LSTMCell(L1, state_is_tuple=True)
            cell_bw = tf.contrib.rnn.LSTMCell(L1, state_is_tuple=True)
            (output_src_fw, output_src_bw), (last_src_fw, last_src_bw) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, self.embed_src, sequence_length=self.len_src, dtype=tf.float32)
            ### for alignment
            self.out_src = tf.concat([output_src_fw, output_src_bw], axis=2)
            self.out_src = tf.nn.dropout(self.out_src, keep_prob=KEEP)
            ### for sentence similarity
            if self.config.sim == 'last':
                self.snt_src = tf.concat([last_src_fw[1], last_src_bw[1]], axis=1)
            elif self.config.sim == 'max':
                mask = tf.sequence_mask(self.len_src, maxlen=tf.shape(self.out_src)[1])
                self.snt_src = tf.reduce_max(tf.boolean_mask(self.out_src, mask), axis=1)
            elif self.config.sim == 'mean':
                mask = tf.sequence_mask(self.len_src, maxlen=tf.shape(self.out_src)[1])
                self.snt_src = tf.reduce_sum(self.out_src, axis=1) / self.len_src
            else:
                sys.stderr.write("error: bad -sim option '{}'\n".format(self.config.sim))
                sys.exit()

            self.snt_src = tf.nn.l2_normalize(self.snt_src, dim=1) 

#        sys.stderr.write("Total src parameters: {}\n".format(sum(variable.get_shape().num_elements() for variable in tf.trainable_variables())))

        ###
        ### tgt-side
        ###
        NW = self.config.tgt_voc_size #tgt vocab
        ES = self.config.tgt_emb_size #tgt embedding size
        L1 = self.config.tgt_lstm_size #tgt lstm size
        #print("TGT NW={} ES={}".format(NW,ES))
        with tf.device('/cpu:0'), tf.variable_scope("embedding_tgt"):
            self.LT_tgt = tf.get_variable(initializer = self.embedding_initialize(NW, ES, self.config.emb_tgt), dtype=tf.float32, name="embeddings_tgt")
            self.embed_tgt = tf.nn.embedding_lookup(self.LT_tgt, self.input_tgt, name="embed_tgt")
            self.embed_tgt = tf.nn.dropout(self.embed_tgt, keep_prob=KEEP)
            
        with tf.variable_scope("lstm_tgt"):
            #print("TGT L1={}".format(L1))
            cell_fw = tf.contrib.rnn.LSTMCell(L1, state_is_tuple=True)
            cell_bw = tf.contrib.rnn.LSTMCell(L1, state_is_tuple=True)
            (output_tgt_fw, output_tgt_bw), (last_tgt_fw, last_tgt_bw) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, self.embed_tgt, sequence_length=self.len_tgt, dtype=tf.float32)
            ### for alignment
            self.out_tgt = tf.concat([output_tgt_fw, output_tgt_bw], axis=2)
            self.out_tgt = tf.nn.dropout(self.out_tgt, keep_prob=KEEP)
            ### for sentence similarity
            if self.config.sim == 'last':
                self.snt_tgt = tf.concat([last_tgt_fw[1], last_tgt_bw[1]], axis=1)
            elif self.config.sim == 'max':
                mask = tf.sequence_mask(self.len_tgt, maxlen=tf.shape(self.out_tgt)[1])
                self.snt_tgt = tf.reduce_max(mask * self.out_tgt, axis=1)
            elif self.config.sim == 'mean':
                mask = tf.sequence_mask(self.len_tgt, maxlen=tf.shape(self.out_tgt)[1])
                self.snt_tgt = tf.reduce_sum(self.out_tgt, axis=1) / self.len_tgt
            else:
                sys.stderr.write("error: bad -sim option '{}'\n".format(self.config.sim))
                sys.exit()

            self.snt_tgt = tf.nn.l2_normalize(self.snt_tgt, dim=1) 

#        sys.stderr.write("Total src/tgt parameters: {}\n".format(sum(variable.get_shape().num_elements() for variable in tf.trainable_variables())))
#        for variable in tf.trainable_variables():
#            sys.stderr.write("var {} params={}\n".format(variable,variable.get_shape().num_elements()))

        with tf.name_scope("similarity"):
            # next is a tensor containing similarity distances (one for each sentence pair) using the last vectors
            self.cos_similarity = tf.reduce_sum(tf.nn.l2_normalize(self.snt_src, dim=1) * tf.nn.l2_normalize(self.snt_tgt, dim=1), axis=1) ### +1:similar -1:divergent

        with tf.name_scope("align"):
            self.align = tf.map_fn(lambda (x,y): tf.matmul(x,tf.transpose(y)), (self.out_src, self.out_tgt), dtype = tf.float32, name="align")

        with tf.name_scope("aggregation"):
            ###
            ### for each src (or tgt) word aggregate the errors of reference/predicted alignments to tgt (or src) words
            ###
            align_ones_mask = tf.greater(self.align, tf.zeros_like(self.align,dtype=tf.float32))
            input_ali_mask = tf.equal(self.input_ali, 1.0+tf.zeros_like(self.align,dtype=tf.float32)) ### all to 0.0 when inference (no alignment given)
            ones_mask = tf.to_float(tf.logical_or(align_ones_mask,input_ali_mask)) ### this is the mask of the words for which i compute the error

            if self.config.error == 'exp':
                error_ones =  tf.log(1 + tf.exp(self.align * -self.input_ali)) * ones_mask ### do not consider errors of words not predicted aligned or not aligned in the reference
            elif self.config.error == 'mse':
                error_ones =  tf.pow(self.align - self.input_ali,2) * ones_mask
            else: 
                sys.stderr.write("error: bad -error option '{}'\n".format(self.config.error))
                sys.exit()

            self.error_src = tf.map_fn(lambda (x,l) : tf.reduce_sum(x[:l,:],0), (tf.transpose(error_ones,[0,2,1]), self.len_tgt), dtype=tf.float32, name="error_src")
            self.error_tgt = tf.map_fn(lambda (x,l) : tf.reduce_sum(x[:l,:],0), (error_ones,                       self.len_src), dtype=tf.float32, name="error_tgt")

            pred_ones = self.align * tf.to_float(align_ones_mask) ### matrix that contain the alignment predictions only if positive (aligned pair)
            ### aggr is sum over all predicted aligned
            self.aggregation_src = tf.map_fn(lambda (x,l) : tf.reduce_sum(x[:l,:],0), (tf.transpose(pred_ones,[0,2,1]), self.len_tgt), dtype=tf.float32, name="aggregation_src")
            self.aggregation_tgt = tf.map_fn(lambda (x,l) : tf.reduce_sum(x[:l,:],0), (pred_ones,                       self.len_src), dtype=tf.float32, name="aggregation_tgt")


#            if self.config.aggr == 'sum':
#                self.aggregation_src = tf.map_fn(lambda (x,l) : tf.reduce_sum(x[:l,:],0), (tf.transpose(error_ones,[0,2,1]), self.len_tgt), dtype=tf.float32, name="aggregation_src")
#                self.aggregation_tgt = tf.map_fn(lambda (x,l) : tf.reduce_sum(x[:l,:],0), (error_ones,                       self.len_src), dtype=tf.float32, name="aggregation_tgt")
#            elif self.config.aggr == 'lse':
#                R = 1.0
#                self.aggregation_src = tf.divide(tf.log(tf.map_fn(lambda (x,l) : tf.reduce_sum(x[:l,:],0), (tf.exp(tf.transpose(self.error_ones,[0,2,1]) * R), self.len_tgt), dtype=tf.float32)), R, name="aggregation_src")
#                self.aggregation_tgt = tf.divide(tf.log(tf.map_fn(lambda (x,l) : tf.reduce_sum(x[:l,:],0), (tf.exp(self.error_ones * R),                       self.len_src), dtype=tf.float32)), R, name="aggregation_tgt")
#            else: 
#                sys.stderr.write("error: bad -aggr option '{}'\n".format(self.config.aggr))
#                sys.exit()

    def add_loss(self):
        with tf.name_scope("loss"):

            self.loss_src = tf.reduce_mean(tf.map_fn(lambda (x,l): tf.reduce_sum(x[:l]), (self.error_src, self.len_src), dtype=tf.float32))
            self.loss_tgt = tf.reduce_mean(tf.map_fn(lambda (x,l): tf.reduce_sum(x[:l]), (self.error_tgt, self.len_tgt), dtype=tf.float32))
            self.loss = self.loss_tgt + self.loss_src

#            if self.config.mode == 'mse' or self.config.mode == 'exp':
#                if self.config.mode == 'mse': 
#                    self.error = tf.pow(self.align - self.input_ali, 2) ### mean squared error of alignment pairs
#                elif self.config.mode == 'exp': 
#                    self.error = tf.log(1 + tf.exp(self.align * -self.input_ali)) ### LogExp error of alignment pairs
#                ### scale errors of aligned pairs (input_ali==1.0)
#                self.error = tf.where(tf.equal(self.input_ali,1.0), self.error*self.scale, self.error)            
#                ### loss as average of individual errors
#                self.loss = tf.reduce_sum(self.error) / tf.count_nonzero(self.input_ali, dtype = tf.float32)
#            elif self.config.mode == 'agg': 

    def add_train(self):
        if   self.config.lr_method == 'adam':     optimizer = tf.train.AdamOptimizer() #self.lr)
        elif self.config.lr_method == 'adagrad':  optimizer = tf.train.AdagradOptimizer(self.lr)
        elif self.config.lr_method == 'sgd':      optimizer = tf.train.GradientDescentOptimizer(self.lr)
        elif self.config.lr_method == 'rmsprop':  optimizer = tf.train.RMSPropOptimizer(self.lr)
        elif self.config.lr_method == 'adadelta': optimizer = tf.train.AdadeltaOptimizer(self.lr)
        else:
            sys.stderr.write("error: bad -lr_method option '{}'\n".format(self.config.lr_method))
            sys.exit()

        if self.config.clip > 0.0:
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), self.config.clip)
            self.train_op = optimizer.apply_gradients(zip(grads, tvars))
        else:
            self.train_op = optimizer.minimize(self.loss)


    def build_graph(self):
        self.add_placeholders()
        self.add_model()  
        if self.config.tst is None: 
            self.add_loss()
            self.add_train()

###################
### feed_dict #####
###################

    def get_feed_dict(self, src, tgt, ali, len_src, len_tgt, lr):
        feed = { 
            self.input_src: src,
            self.input_tgt: tgt,
            self.input_ali: ali,
            self.len_src: len_src,
            self.len_tgt: len_tgt,
            self.lr: lr
        }
        return feed

###################
### learning ######
###################

    def run_epoch(self, train, dev, lr):

        #######################
        # learn on trainset ###
        #######################
        nbatches = (len(train) + self.config.batch_size - 1) // self.config.batch_size
        curr_epoch = self.config.last_epoch + 1
        TLOSS = 0.0 # training loss
        ILOSS = 0.0 # intermediate loss (average over [config.report_every] iterations)
        tscore = Score()
        iscore = Score()
        ini_time = time.time()
        for iter, (src_batch, tgt_batch, ali_batch, raw_src_batch, raw_tgt_batch, len_src_batch, len_tgt_batch) in enumerate(minibatches(train, self.config.batch_size)):
            fd = self.get_feed_dict(src_batch, tgt_batch, ali_batch, len_src_batch, len_tgt_batch, lr)
#            print("raw_src_batch is \n{}".format(raw_src_batch))
#            print("raw_tgt_batch is \n{}".format(raw_tgt_batch))
#            print("src_batch is \n{}".format(src_batch))
#            print("tgt_batch is \n{}".format(tgt_batch))
#            print("ali_batch is \n{}".format(ali_batch))
#            print("len_src_batch is \n{}".format(len_src_batch))
#            print("len_tgt_batch is \n{}".format(len_tgt_batch))
            _, loss, align = self.sess.run([self.train_op, self.loss, self.align], feed_dict=fd)
#            print("align is \n{}".format(align))
#            print("error is \n{}".format(error))
#            print("loss is \n{}".format(loss))
#            sys.exit()
            tscore.add_batch(align, ali_batch)
            iscore.add_batch(align, ali_batch)
            TLOSS += loss
            ILOSS += loss

            if (iter+1)%self.config.report_every == 0:
                curr_time = time.strftime("[%Y-%m-%d_%X]", time.localtime())
                iscore.summarize()
                ILOSS = ILOSS/self.config.report_every
                sys.stderr.write('{} Epoch {} Iteration {}/{} loss:{:.4f} ({})\n'.format(curr_time,curr_epoch,iter+1,nbatches,ILOSS,iscore.results))
                ILOSS = 0.0
                iscore = Score()

        TLOSS = TLOSS/nbatches
        tscore.summarize()
        curr_time = time.strftime("[%Y-%m-%d_%X]", time.localtime())
        sys.stderr.write('{} Epoch {} TRAIN loss={:.4f} ({}) lr={:.4f}'.format(curr_time,curr_epoch,TLOSS,tscore.results,lr))
        unk_src = float(100) * train.nunk_src / train.nsrc
        unk_tgt = float(100) * train.nunk_tgt / train.ntgt
        sys.stderr.write(' Train set: words={}/{} %ones={:.2f} pair={} unpair={} swap={} extend={} replace={} %unk={:.2f}/{:.2f}\n'.format(train.nsrc,train.ntgt,100.0*train.nones/train.nlnks,train.npair,train.nunpair,train.nswap,train.nextend,train.nreplace,unk_src,unk_tgt))

        ##########################
        # evaluate over devset ###
        ##########################
        VLOSS = 0.0
        if dev is not None:
            nbatches = (len(dev) + self.config.batch_size - 1) // self.config.batch_size
            # iterate over dataset
            VLOSS = 0
            vscore = Score()

            for iter, (src_batch, tgt_batch, ali_batch, raw_src_batch, raw_tgt_batch, len_src_batch, len_tgt_batch) in enumerate(minibatches(dev, self.config.batch_size)):
                fd = self.get_feed_dict(src_batch, tgt_batch, ali_batch, len_src_batch, len_tgt_batch, 0.0)
                loss, align = self.sess.run([self.loss, self.align], feed_dict=fd)
                vscore.add_batch(align, ali_batch)
                VLOSS += loss # append single value which is a mean of losses of the n examples in the batch
            vscore.summarize()
            VLOSS = VLOSS/nbatches
            sys.stderr.write('{} Epoch {} VALID loss={:.4f} ({})'.format(curr_time,curr_epoch,VLOSS,vscore.results))
            unk_src = float(100) * dev.nunk_src / dev.nsrc
            unk_tgt = float(100) * dev.nunk_tgt / dev.ntgt
            sys.stderr.write(' Valid set: words={}/{} %ones={:.2f} pair={} unpair={} swap={} extend={} replace={} %unk={:.2f}/{:.2f}\n'.format(dev.nsrc,dev.ntgt,100.0*dev.nones/dev.nlnks,dev.npair,dev.nunpair,dev.nswap,dev.nextend,dev.nreplace,unk_src,unk_tgt,VLOSS))

        #################################
        #keep record of current epoch ###
        #################################
        self.config.tloss = TLOSS
        self.config.tA = tscore.A
        self.config.tP = tscore.P
        self.config.tR = tscore.R
        self.config.tF = tscore.F
        self.config.time = time.strftime("[%Y-%m-%d_%X]", time.localtime())
        self.config.seconds = "{:.2f}".format(time.time() - ini_time)
        self.config.last_epoch += 1
        self.save_session(self.config.last_epoch)
        if dev is not None:
            self.config.vloss = VLOSS
            self.config.vA = vscore.A
            self.config.vP = vscore.P
            self.config.vR = vscore.R
            self.config.vF = vscore.F
        self.config.write_config()
        return VLOSS, curr_epoch


    def learn(self, train, dev, n_epochs):
        lr = self.config.lr
        curr_time = time.strftime("[%Y-%m-%d_%X]", time.localtime())
        sys.stderr.write("{} Training with {} sentence pairs: {} batches with up to {} examples each.\n".format(curr_time,len(train),(len(train)+self.config.batch_size-1)//self.config.batch_size,self.config.batch_size))
        best_score = 0
        best_epoch = 0
        for iter in range(n_epochs):
            score, epoch = self.run_epoch(train, dev, lr)  ### decay when score does not improve over the best
            curr_time = time.strftime("[%Y-%m-%d_%X]", time.localtime())
            if iter == 0 or score <= best_score: 
                best_score = score
                best_epoch = epoch
            else:
                lr *= self.config.lr_decay # decay learning rate

###################
### inference #####
###################

    def inference(self, tst):

        if self.config.show_svg: print "<html>\n<body>"
        nbatches = (len(tst) + self.config.batch_size - 1) // self.config.batch_size
        score = Score()
        n_sents = 0

        for iter, (src_batch, tgt_batch, ali_batch, raw_src_batch, raw_tgt_batch, len_src_batch, len_tgt_batch) in enumerate(minibatches(tst, self.config.batch_size)):
            fd = self.get_feed_dict(src_batch, tgt_batch, ali_batch, len_src_batch, len_tgt_batch, 0.0)

            align_batch, snt_src_batch, snt_tgt_batch, sim_batch, aggr_src_batch, aggr_tgt_batch = self.sess.run([self.align, self.snt_src, self.snt_tgt, self.cos_similarity, self.aggregation_src, self.aggregation_tgt], feed_dict=fd)
            if tst.annotated: 
                score.add_batch(align_batch, ali_batch)

            for i_sent in range(len(align_batch)):
                n_sents += 1
                v = Visualize(n_sents,raw_src_batch[i_sent],raw_tgt_batch[i_sent],sim_batch[i_sent],align_batch[i_sent],aggr_src_batch[i_sent],aggr_tgt_batch[i_sent],snt_src_batch[i_sent],snt_tgt_batch[i_sent])
                if self.config.show_svg: v.print_svg()
                elif self.config.show_matrix: v.print_matrix()
                else: v.print_vectors(self.config.show_sim,self.config.show_align)

        if tst.annotated:
            score.summarize()
            unk_s = float(100) * tst.nunk_src / tst.nsrc
            unk_t = float(100) * tst.nunk_tgt / tst.ntgt
            curr_time = time.strftime("[%Y-%m-%d_%X]", time.localtime())
            sys.stderr.write('{} TEST ({})'.format(curr_time,score.results))
            sys.stderr.write(' Test set: words={}/{} %ones={:.2f} pair={} unpair={} swap={} extend={} replace={} %unk={:.2f}/{:.2f}\n'.format(tst.nsrc,tst.ntgt,100.0*tst.nones/tst.nlnks,tst.npair,tst.nunpair,tst.nswap,tst.nextend,tst.nreplace,unk_src,unk_tgt,VLOSS))

            sys.stderr.write('TEST words={}/{} %unk={:.2f}/{:.2f} ({})\n'.format(tst.nsrc,tst.ntgt,unk_s,unk_t,score.results))

        if self.config.show_svg: print "</body>\n</html>"


###################
### session #######
###################

    def initialize_session(self):
        self.sess = tf.Session()
        self.saver = tf.train.Saver(max_to_keep=20)

        if self.config.epoch is not None: ### restore a file for testing
            fmodel = self.config.mdir + '/epoch' + self.config.epoch
            sys.stderr.write("Restoring model: {}\n".format(fmodel))
            self.saver.restore(self.sess, fmodel)
            return

        if self.config.mdir: ### initialize for training or restore previous
            if not os.path.exists(self.config.mdir + '/checkpoint'): 
                sys.stderr.write("Initializing model\n")
                self.sess.run(tf.global_variables_initializer())
            else:
                sys.stderr.write("Restoring previous model: {}\n".format(self.config.mdir))
                self.saver.restore(self.sess, tf.train.latest_checkpoint(self.config.mdir))

    def save_session(self,e):
        if not os.path.exists(self.config.mdir): os.makedirs(self.config.mdir)
        file = "{}/epoch{}".format(self.config.mdir,e)
        self.saver.save(self.sess, file) #, max_to_keep=4, write_meta_graph=False) # global_step=step, keep_checkpoint_every_n_hours=2

    def close_session(self):
        self.sess.close()


