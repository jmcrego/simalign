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

class Visualize():

    def __init__(self,n_sents,isrc,itgt,src,tgt,sim,align,aggr_src,aggr_tgt,last_src,last_tgt,mark_unks=False): 
        self.n_sents = n_sents
        self.isrc = isrc
        self.itgt = itgt
        self.src = src
        self.tgt = tgt
        self.sim = sim
        self.align = align
        self.aggr_src = aggr_src
        self.aggr_tgt = aggr_tgt
        self.last_src = last_src
        self.last_tgt = last_tgt

        if mark_unks:
            for s in range(len(self.isrc)):
                if self.isrc[s]==0: 
                    self.src[s] = '@@@'+self.src[s]+'@@@'
            for t in range(len(self.itgt)):
                if self.itgt[t]==0: 
                    self.tgt[t] = '@@@'+self.tgt[t]+'@@@'


    def print_matrix(self):
        print('<:::{}:::> cosine sim = {:.4f}'.format(self.n_sents, self.sim))
        source = list(self.src)
        target = list(self.tgt)

        max_length_tgt_tokens = max(5,max([len(x) for x in target]))
        A = str(max_length_tgt_tokens+1)
        print(''.join(("{:"+A+"}").format(t) for t in target))
        for s in range(len(source)):
            for t in range(len(target)):
                myscore = "{:+.2f}".format(self.align[s][t])
                while len(myscore) < max_length_tgt_tokens+1: myscore += ' '
                sys.stdout.write(myscore)
            print(source[s])

    def print_svg(self):
        start_x = 25
        start_y = 100
        len_square = 15
        len_x = len(self.tgt)
        len_y = len(self.src)
        separation = 2
        print "<br>\n<svg width=\""+str(len_x*len_square + start_x + 150)+"\" height=\""+str(len_y*len_square + start_y)+"\">"
        for x in range(len(self.tgt)): ### tgt
            col="black"
            print "<text x=\""+str(x*len_square + start_x)+"\" y=\""+str(start_y-2)+"\" fill=\""+col+"\" font-family=\"Courier\" font-size=\"5\">"+"{:+.1f}".format(self.aggr_tgt[x])+"</text>"
            col="black" ### remove this line if you want divergent words in red
            print "<text x=\""+str(x*len_square + start_x + separation)+"\" y=\""+str(start_y-15)+"\" fill=\""+col+"\" font-family=\"Courier\" font-size=\"10\" transform=\"rotate(-45 "+str(x*len_square + start_x + 10)+","+str(start_y-15)+") \">"+self.tgt[x]+"</text>"

        maxlnk = np.max(self.align)
        for y in range(len(self.src)): ### src
            for x in range(len(self.tgt)): ### tgt
                lnk = self.align[y][x]
                color = 256 #white (not aligned) 0 is black
                if lnk > 0: #aligned
                    p_lnk = lnk/maxlnk #probability of link
                    p = 1-p_lnk #probability of not link
                    color = int(256*p)
                print "<rect x=\""+str(x*len_square + start_x)+"\" y=\""+str(y*len_square + start_y)+"\" width=\""+str(len_square)+"\" height=\""+str(len_square)+"\" style=\"fill:rgb("+str(color)+","+str(color)+","+str(color)+"); stroke-width:1;stroke:rgb(200,200,200)\" />"
                txtcolor = "black"
                if self.align[y][x] < 0: txtcolor="red"
                print "<text x=\""+str(x*len_square + start_x)+"\" y=\""+str(y*len_square + start_y + len_square*3/4)+"\" fill=\"{}\" font-family=\"Courier\" font-size=\"5\">".format(txtcolor)+"{:+.1f}".format(self.align[y][x])+"</text>"

            col="black"
            print "<text x=\""+str(len_x*len_square + start_x + separation)+"\" y=\""+str(y*len_square + start_y + len_square*3/4)+"\" fill=\""+col+"\" font-family=\"Courier\" font-size=\"5\">"+"{:+.1f}".format(self.aggr_src[y])+"</text>"
            col="black" ### remove this line if you want divergent words in red
            print "<text x=\""+str(len_x*len_square + start_x + separation + 15)+"\" y=\""+str(y*len_square + start_y + len_square*3/4)+"\" fill=\""+col+"\" font-family=\"Courier\" font-size=\"10\">"+self.src[y]+"</text>"
        print("<br>\n<svg width=\"200\" height=\"20\">")
        print("<text x=\"{}\" y=\"10\" fill=\"black\" font-family=\"Courier\" font-size=\"8\"\">{:+.4f}</text>".format(start_x,self.sim))

    def print_vectors(self,show_last,show_align):
        line = []
        line.append("{:.4f}".format(self.sim))
#        if show_aggr_sim:
#            aggr = (sum(self.aggr_src) + sum(self.aggr_tgt)) / np.float32(len(self.src) + len(self.tgt))
#            line.append("{:.4f}".format(aggr))
        line.append(" ".join(s for s in self.src))
        line.append(" ".join(t for t in self.tgt))

        if show_last:
            line.append(" ".join("{:.4f}".format(s) for s in self.last_src))
            line.append(" ".join("{:.4f}".format(t) for t in self.last_tgt))

        if show_align: 
            matrix = []
            for s in range(len(self.src)):
                row = " ".join("{:.4f}".format(self.align[s,t]) for t in range(len(self.tgt)))
                matrix.append(row)
            line.append("\t".join(row for row in matrix))        

        print "\t".join(line)


