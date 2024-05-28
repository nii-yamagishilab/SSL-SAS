#!/usr/bin/env python
""" Compute cosince distance between
"""
import sys
from kaldiio import WriteHelper, ReadHelper
import os
from os.path import join, isdir

from scipy.spatial import distance

args = sys.argv

src_xvec_file = args[1]
pool_xvec_file = args[2]
scores_dir = args[3]

if not isdir(scores_dir):
    os.makedirs(scores_dir)

pool_xvectors = {}
c = 0
with ReadHelper('scp:'+pool_xvec_file) as reader:
    for key, xvec in reader:
        #print key, mat.shape
        pool_xvectors[key] = xvec
        c += 1
print("Read ", c, "pool xvectors")

with ReadHelper('scp:'+src_xvec_file) as reader:
    for sspk, sxvec in reader:
        print("Computing cosine measure for " + sspk)
        with open(join(scores_dir, 'affinity_'+sspk), 'w') as sf:
            for pspk, pxvec in pool_xvectors.items():
                # compute cosine distance between src and pool spk
                # Multiplying by -1 to ensure compatibility with affinity
                # Now lower value will indicate less affinity as compared
                # to original cosine distance
                dist = -1.0 * distance.cosine(sxvec.squeeze(), pxvec.squeeze())
                sf.write(sspk + ' ' + pspk + ' ' + str(dist) + '\n')

