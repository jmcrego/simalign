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

    def __init__(self,n_sents,src,tgt,sim,align): 
        self.n_sents = n_sents
        self.src = src
        self.tgt = tgt
        self.sim = sim
        self.align = align


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
            print "<text x=\""+str(x*len_square + start_x)+"\" y=\""+str(start_y-2)+"\" fill=\""+col+"\" font-family=\"Courier\" font-size=\"5\">"+"</text>"
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
            print "<text x=\""+str(len_x*len_square + start_x + separation)+"\" y=\""+str(y*len_square + start_y + len_square*3/4)+"\" fill=\""+col+"\" font-family=\"Courier\" font-size=\"5\">"+"</text>"
            col="black" ### remove this line if you want divergent words in red
            print "<text x=\""+str(len_x*len_square + start_x + separation + 15)+"\" y=\""+str(y*len_square + start_y + len_square*3/4)+"\" fill=\""+col+"\" font-family=\"Courier\" font-size=\"10\">"+self.src[y]+"</text>"
        print("<br>\n<svg width=\"200\" height=\"20\">")
        print("<text x=\"{}\" y=\"10\" fill=\"black\" font-family=\"Courier\" font-size=\"8\"\">{:+.4f}</text>".format(start_x,self.sim))

    def print_vectors(self,last_src,last_tgt,align):
        line = []
        line.append("{:.4f}".format(self.sim))
        line.append(" ".join(s for s in self.src))
        line.append(" ".join(t for t in self.tgt))

        if len(last_src) and len(last_tgt): 
            line.append(" ".join("{:.4f}".format(s) for s in last_src))
            line.append(" ".join("{:.4f}".format(t) for t in last_tgt))

        if len(align): 
            matrix = []
            for s in range(len(self.src)):
                row = " ".join("{:.4f}".format(align[s,t]) for t in range(len(self.tgt)))
                matrix.append(row)
            line.append("\t".join(row for row in matrix))
        
        print "\t".join(line)
