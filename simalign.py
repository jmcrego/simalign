# -*- coding: utf-8 -*-
import tensorflow as tf
import os
import sys
from dataset import Dataset, Vocab
from model import Model
from config import Config
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)

def main():
    config = Config(sys.argv)
    model = Model(config)
    model.build_graph()
    model.initialize_session()

    if config.trn:
        trn = Dataset(config.trn, config.voc_src, config.voc_tgt, config.seq_size, config.max_sents, config.p_unpair, config.p_swap, config.p_remove, config.p_extend, config.p_replace, do_shuffle=True)
        dev = Dataset(config.dev, config.voc_src, config.voc_tgt, seq_size=0,      max_sents=0,      p_unpair=0.0,    p_swap=0.0,    p_remove=0.0,    p_extend=0.0,    p_replace=0.0,    do_shuffle=False)
        model.learn(trn, dev, config.n_epochs)
    if config.tst:
        tst = Dataset(config.tst, config.voc_src, config.voc_tgt, seq_size=0,      max_sents=0,      p_unpair=0.0,    p_swap=0.0,    p_remove=0.0   , p_extend=0.0,    p_replace=0.0,    do_shuffle=False)
        model.inference(tst)

    model.close_session()

if __name__ == "__main__":
    main()
