# -*- coding: utf-8 -*-
import os.path
import io
from math import *
from random import shuffle
import random
import numpy as np
import sys
import time
import gzip
from collections import defaultdict
reload(sys)
sys.setdefaultencoding('utf8')

idx_unk = 0
str_unk = "<unk>"

idx_pad = 1
str_pad = "<pad>"

class Embeddings():

    def __init__(self, file, voc, length):
        w2e = {}
        if file is not None:
            #with io.open(file, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
            if file.endswith('.gz'): f = gzip.open(file, 'rb')
            else: f = io.open(file, 'r', encoding='utf-8', newline='\n', errors='ignore')

            self.num, self.dim = map(int, f.readline().split())
            i = 0
            for line in f:
                i += 1
                if i%10000 == 0:
                    if i%100000 == 0: sys.stderr.write("{}".format(i))
                    else: sys.stderr.write(".")
                tokens = line.rstrip().split(' ')
                if voc.exists(tokens[0]): w2e[tokens[0]] = tokens[1:] 
            f.close()
            sys.stderr.write('Read {} embeddings ({} missing in voc)\n'.format(len(w2e),len(voc)-len(w2e)))
        else:
            sys.stderr.write('Embeddings file not used! will be initialised to [{}x{}]\n'.format(len(voc),length))
            self.dim = length

        # i need an embedding for each word in voc
        # embedding matrix must have tokens in same order than voc 0:<unk>, 1:<pad>, 2:le, ...
        self.matrix = []
        for tok in voc:
            if tok == str_unk or tok == str_pad or not tok in w2e: ### random initialize these tokens
                self.matrix.append(np.random.normal(0, 1.0, self.dim)) 
            else:
                self.matrix.append(np.asarray(w2e[tok], dtype=np.float32))

        self.matrix = np.asarray(self.matrix, dtype=np.float32)
        self.matrix = self.matrix / np.sqrt((self.matrix ** 2).sum(1))[:, None]

class Vocab():

    def __init__(self, dict_file):
        self.tok_to_idx = {}
        self.idx_to_tok = []
        self.idx_to_tok.append(str_unk)
        self.tok_to_idx[str_unk] = len(self.tok_to_idx) #0
        self.idx_to_tok.append(str_pad)
        self.tok_to_idx[str_pad] = len(self.tok_to_idx) #1
        nline = 0
        with io.open(dict_file, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
            for line in f:
#        for line in [line.rstrip('\n') for line in open(dict_file)]:
                nline += 1
                line = line.strip()
                self.idx_to_tok.append(line)
                self.tok_to_idx[line] = len(self.tok_to_idx)

        self.length = len(self.idx_to_tok)
        sys.stderr.write('Read vocab ({} entries)\n'.format(self.length))

    def __len__(self):
        return len(self.idx_to_tok)

    def __iter__(self):
        for tok in self.idx_to_tok:
            yield tok

    def exists(self, s):
        return s in self.tok_to_idx

    def get(self,s):
        if type(s) == int: ### I want the string
            if s < len(self.idx_to_tok): return self.idx_to_tok[s]
            else:
                sys.stderr.write('error: key \'{}\' not found in vocab\n'.format(s))
                sys.exit()
        ### I want the index
        if s not in self.tok_to_idx: return idx_unk
        return self.tok_to_idx[s]


class Dataset():

    def __init__(self, file, voc_src, voc_tgt, seq_size, max_sents, p_unpair, p_swap, p_remove, p_extend, p_replace, do_shuffle):
        if file is None: return None
        self.voc_src = voc_src 
        self.voc_tgt = voc_tgt 
        self.file = file
        self.seq_size = seq_size
        self.max_sents = max_sents
        self.do_shuffle = do_shuffle
        self.annotated = False
        self.p_unpair = p_unpair
        self.p_swap = p_swap
        self.p_remove = p_remove
        self.p_extend = p_extend
        self.p_replace = p_replace
        self.data = []
        self.length = 0 ### length of the data set to be used (not necessarily the whole set)
        self.max_rep = 100

#        with io.open(self.file, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        if self.file.endswith('.gz'): f = gzip.open(self.file, 'rb')
        else: f = io.open(self.file, 'r', encoding='utf-8', newline='\n', errors='ignore')
        nline = 0
        for line in f:
            line = line.strip('\n') 
            nline += 1
            ntokens = len(line.split('\t'))
            if nline==1 and ntokens==3: self.annotated = True
            if (self.annotated and ntokens != 3) or (not self.annotated and ntokens != 2):
                sys.stderr.write("warning: bad data entry \'{}\' in line={} [skipped]\n".format(line,nline))
                continue;
            tokens = line.split('\t')
            src = tokens[0].split(' ')
            tgt = tokens[1].split(' ')
            if self.seq_size > 0 and (len(src) > self.seq_size or len(tgt) > self.seq_size): continue
            ali = []
            if ntokens == 3: ali = tokens[2].split(' ')
            self.data.append([src,tgt,ali])
            self.length += 1
        f.close()

        if self.max_sents > 0:
            self.length = min(self.length,self.max_sents)
        sys.stderr.write('({} contains {} examples)\n'.format(self.file,len(self.data)))

    def __len__(self):
        return self.length

    def similar_size(self, x, y):
        m = min(x,y) 
        M = max(x,y)
        d = M - m
        #m=[1,4] => d<4
        #m=[5,9] => d<5
        #m=[10,14] => d<6
        return d < int(m/5)+4

    def get_unpair_example(self, index):
        (src, tgt, ali) = self.data[index]
        n_rep = 0
        while True:
            n_rep += 1
            if n_rep > self.max_rep: break
            index2 = int(random.random()*len(self.data))
            if index2 == index: continue
            (src2, tgt2, ali2) = self.data[index2]
            if random.random() < 0.5: #replace src by src2 
                if self.similar_size(len(src2),len(tgt)): return src2, tgt, []
            else: #replace tgt by tgt2
                if self.similar_size(len(src),len(tgt2)): return src, tgt2, []
        return [], [], []

    def get_extend_example(self, index):
        (src, tgt, ali) = self.data[index]
        n_rep = 0
        while True:
            n_rep += 1
            if n_rep > self.max_rep: break
            index2 = int(random.random()*len(self.data))
            (src2, tgt2, ali2) = self.data[index2]
            if len(src2)>=5 and len(tgt2)>=5:
                if random.random() < 0.5: #extend in src
                    i = int(random.random() * (len(src2)-3)) + 2
                    src.extend(src2[i:])
                else: #extend in tgt
                    i = int(random.random() * (len(tgt2)-3)) + 2
                    tgt.extend(tgt2[i:]) 
                return src, tgt, ali
        return [], [], []

    def get_replace_example(self, index):
        (src, tgt, ali) = self.data[index]
        # to replace, sentences must be at least 10 words
        if len(src) < 10 or len(tgt) < 10: return [], [], []
        n_rep = 0
        while True:
            n_rep += 1
            if n_rep > self.max_rep: break
    
            if random.random() < 0.5: #replace in src
                ini = int(random.random() * len(src)) 
                l = int(random.random() * (len(src)-ini-1)) + 1
                end = ini + l
                index2 = int(random.random()*len(self.data))
                (src2, tgt2, _) = self.data[index2]
                if len(src2) < l: continue
                ini2 = int(random.random() * (len(src2)-l))
                end2 = ini2 + l
                for s in range(ini,end): src[s] = src2[s+ini2-ini]
                ali2 = []
                for a in ali:
                    if len(a.split('-')) != 2:
                        sys.stderr.write('warning: bad alignment: {}\n'.format(a))
                        continue
                    s, t = map(int, a.split('-'))
                    if s >= len(src):
                        sys.stderr.write('warning: src alignment: {} out of bounds: {}\n'.format(s, src))
                        continue
                    if t >= len(tgt):
                        sys.stderr.write('warning: tgt alignment: {} out of bounds: {}\n'.format(t, tgt))
                        continue
                    if (s<ini or s>=end): ali2.append("{}-{}".format(s,t))
                return src, tgt, ali2

            else: #replace in tgt
                ini = int(random.random() * len(tgt)) 
                l = int(random.random() * (len(tgt)-ini-1)) + 1
                end = ini + l
                index2 = int(random.random()*len(self.data))
                (src2, tgt2, _) = self.data[index2]
                if len(tgt2) < l: continue
                ini2 = int(random.random() * (len(tgt2)-l))
                end2 = ini2 + l
                for t in range(ini,end): tgt[t] = tgt2[t+ini2-ini]
                ali2 = []
                for a in ali:
                    if len(a.split('-')) != 2:
                        sys.stderr.write('warning: bad alignment: {}\n'.format(a))
                        continue
                    s, t = map(int, a.split('-'))
                    if s >= len(src):
                        sys.stderr.write('warning: src alignment: {} out of bounds: {}\n'.format(s,src))
                        continue
                    if t >= len(tgt):
                        sys.stderr.write('warning: tgt alignment: {} out of bounds: {}\n'.format(t,tgt))
                        continue
                    if (t<ini or t>=end): ali2.append("{}-{}".format(s,t))
                return src, tgt, ali2

        return [], [], []

    def get_swap_example(self, index):
        (src, tgt, ali) = self.data[index]
        # to swap, sentences must be at least 10 words
        if len(src) < 10: return [], [], []

        min_t = [-1] * len(src) #min target word aligned to source s
        max_t = [-1] * len(src) #max target word aligned to source s
        for a in ali:
            if len(a.split('-')) != 2:
                sys.stderr.write('warning: bad alignment: {}\n'.format(a))
                continue
            s, t = map(int, a.split('-'))
            if s >= len(src):
                sys.stderr.write('warning: src alignment: {} out of bounds\n'.format(s))
                continue
            if t >= len(tgt):
                sys.stderr.write('warning: tgt alignment: {} out of bounds\n'.format(t))
                continue
            if min_t[s] == -1 or t < min_t[s]: min_t[s] = t
            if max_t[s] == -1 or t > max_t[s]: max_t[s] = t

        mid = len(src)/2
        points = range(mid-3,mid+3)
        shuffle(points)
        for p in points:
            max_t_p = max(max_t[:p]) ### maximum target word t aligned to source words [:p] --> [0,p-1]
            min_t_p = min(min_t[p:]) ### minimum target word t aligned to source words [p:] --> [p,len(s)-1]
            if min_t_p > max_t_p: 
                ### swap src words and alignments
                ### s1 s2 s3 s4 s5 s6 =(p=4)=> s4 s5 s6 s1 s2 s3
                src2 = []
                for s in range(p,len(src)): src2.append(src[s])            
                for s in range(0,p): src2.append(src[s])            
                ali2 = []
                for a in ali:
                    s, t = map(int, a.split('-'))
                    if s>=p: s = s-p
                    else: s = s + len(src) - p
                    ali2.append("{}-{}".format(s,t))
                return src2, tgt, ali2

        return [], [], []

    def __iter__(self):
        nsent = 0
        self.nsrc = 0
        self.ntgt = 0
        self.nunk_src = 0
        self.nunk_tgt = 0
        self.nones = 0
        self.nlnks = 0
        self.npair = 0
        self.nunpair = 0
        self.nswap = 0
        self.nremove = 0
        self.nreplace = 0
        self.nextend = 0
        ### every iteration i get shuffled data examples if do_shuffle
        indexs = [i for i in range(len(self.data))]
        if self.do_shuffle: shuffle(indexs)
        for index in indexs:
            src, tgt, ali = [], [], []
            if self.annotated:
                p = random.random() # p in [0.0, 1.0)
                ###
                ### unpair
                ###
                pmin = 0.0
                pmax = pmin + self.p_unpair
                if p >= pmin and p < pmax: 
                    (src, tgt, ali) = self.get_unpair_example(index)
                    if len(src) and len(tgt): self.nunpair += 1
                ###
                ### extend
                ###
                pmin = pmax
                pmax = pmin + self.p_extend
                if p >= pmin and p < pmax: 
                    (src, tgt, ali) = self.get_extend_example(index)
                    if len(src) and len(tgt): self.nextend += 1
                ###
                ### swap
                ###
                pmin = pmax
                pmax = pmin + self.p_swap
                if p >= pmin and p < pmax: 
                    (src, tgt, ali) = self.get_swap_example(index)
                    if len(src) and len(tgt): self.nswap += 1
                ###
                ### remove
                ###
                pmin = pmax
                pmax = pmin + self.p_remove
                if p >= pmin and p < pmax: 
                    (src, tgt, ali) = self.get_remove_example(index)
                    if len(src) and len(tgt): self.nremove += 1
                ###
                ### replace
                ###
                pmin = pmax
                pmax = pmin + self.p_replace
                if p >= pmin and p < pmax: 
                    (src, tgt, ali) = self.get_replace_example(index)
                    if len(src) and len(tgt): self.nreplace += 1

            ###
            ### pair
            ###
            if len(src)==0 and len(tgt)==0:
                (src, tgt, ali) = self.data[index] 
                if len(src) and len(tgt): self.npair += 1


            self.nones += len(ali)
            self.nlnks += len(src)*len(tgt)

            isrc = []
            for s in src: 
                isrc.append(self.voc_src.get(s))
                if isrc[-1] == idx_unk: self.nunk_src += 1
                self.nsrc += 1

            itgt = []
            for t in tgt: 
                itgt.append(self.voc_tgt.get(t))
                if itgt[-1] == idx_unk: self.nunk_tgt += 1
                self.ntgt += 1

            yield isrc, itgt, ali, src, tgt
            nsent += 1
            if self.max_sents > 0 and nsent >= self.max_sents: break # already generated max_sents examples

def minibatches(data, minibatch_size):
    SRC, TGT, ALI, RAW_SRC, RAW_TGT = [], [], [], [], []
    max_src, max_tgt = 0, 0
    for (src, tgt, ali, raw_src, raw_tgt) in data:
        if len(SRC) == minibatch_size:
            yield build_batch(SRC, TGT, ALI, RAW_SRC, RAW_TGT, max_src, max_tgt)
            SRC, TGT, ALI, RAW_SRC, RAW_TGT = [], [], [], [], []
            max_src, max_tgt = 0, 0
        if len(src) > max_src: max_src = len(src)
        if len(tgt) > max_tgt: max_tgt = len(tgt)
        SRC.append(src)
        TGT.append(tgt)
        ALI.append(ali)
        RAW_SRC.append(raw_src)
        RAW_TGT.append(raw_tgt)

    if len(SRC) != 0:
        yield build_batch(SRC, TGT, ALI, RAW_SRC, RAW_TGT, max_src, max_tgt)

def build_batch(SRC, TGT, ALI, RAW_SRC, RAW_TGT, max_src, max_tgt):
    src_batch, tgt_batch, ali_batch, raw_src_batch, raw_tgt_batch, len_src_batch, len_tgt_batch = [], [], [], [], [], [], []
    ### build: src_batch, pad_src_batch sized of max_src
    batch_size = len(SRC)
    for i in range(batch_size):
        src = list(SRC[i])
        tgt = list(TGT[i])
        raw_src = list(RAW_SRC[i])
        raw_tgt = list(RAW_TGT[i])
        len_src = len(src)
        len_tgt = len(tgt)
        ali = [[-1.0 for x in range(max_tgt)] for y in range(max_src)] #will contain 1.0 (for present links) -1.0 (non present links) 0.0 (padded words)
        for a in ALI[i]:
            if len(a.split('-')) != 2:
                sys.stderr.write('warning: bad alignment: {} in ali {} [skipped alignment]\n'.format(a,ALI[i]))
                continue
            s, t = map(int, a.split('-'))
            if s >= len_src:
                sys.stderr.write('warning: src alignment: {} out of bounds {} [skipped alignment]\n'.format(s,len(SRC[i])))
                sys.exit()
                continue
            if t >= len_tgt:
                sys.stderr.write('warning: tgt alignment: {} out of bounds {} [skipped alignment]\n'.format(t,len(TGT[i])))
                continue
            ali[s][t] = 1.0
        # add padding to have max_src/max_tgt words in all examples of current batch
        while len(src) < max_src: src.append(idx_pad) #<pad>
        while len(tgt) < max_tgt: tgt.append(idx_pad) #<pad>
        ### ali acts as mask (ali==0 for <pad>'s')
        for s in range(max_src):
            for t in range(max_tgt):
                if s >= len_src or t >= len_tgt: 
                    ali[s][t] = 0.0 #ali is zero for <pad> (it was initialized to -1)
        ### add to batches
        src_batch.append(src)
        tgt_batch.append(tgt)
        ali_batch.append(ali)
        raw_src_batch.append(raw_src)
        raw_tgt_batch.append(raw_tgt)
        len_src_batch.append(len_src)
        len_tgt_batch.append(len_tgt)

#    print("raw_src: {}".format(raw_src_batch))
#    print("raw_tgt: {}".format(raw_tgt_batch))
#    print("src: {}".format(src_batch))
#    print("tgt: {}".format(tgt_batch))
#    print("ali: {}".format(ali_batch))
#    print("len_src: {}".format(len_src_batch))
#    print("len_tgt: {}".format(len_tgt_batch))
#    sys.exit()
    return src_batch, tgt_batch, ali_batch, raw_src_batch, raw_tgt_batch, len_src_batch, len_tgt_batch


