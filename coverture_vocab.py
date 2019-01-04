# -*- coding: utf-8 -*-
#!/usr/bin/python -u

import sys
import io
import gzip

lines = [line.rstrip('\n') for line in open(sys.argv[1])]
vocab = set(lines)

f = sys.argv[2]
if f.endswith('.gz'): f = gzip.open(f, 'rb')
else: f = io.open(f, 'rb')

Unk = {}
nunk = 0
nwrd = 0
for line in f: #sys.stdin:
    words = line.rstrip().split()
    for word in words:
        nwrd += 1
        if word not in vocab:
            nunk += 1
            if word in Unk: 
                Unk[word] += 1
            else: 
                Unk[word] = 1

for wrd,frq in sorted(Unk.items(), key=lambda(k,v): v):
    print("{}\t{}".format(wrd,frq))

print("nunk = {} out of {} words ({:.2f}%)".format(nunk,nwrd,100.0*nunk/nwrd))
