#!/usr/bin/env python
""" Compute average of x-vectors as speaker xvectors

$: python speaker_xvector.py spk2utt xvector.scp outputdir output_name_prefix

"""
import os
import sys
from os.path import join, isdir
import numpy as np

import local_lib

def transform_data(data):
    return data

def average_vectors(data_buf):
    ave_buf = np.zeros(transform_data(data_buf[0]).shape)
    # average, online algorithm
    for idx, data in enumerate(data_buf):
        ave_buf = ave_buf + (transform_data(data) - ave_buf) / (idx + 1)
    return ave_buf
    
def collect_data_spk(data_pool_dic, spk2utt_dic, spk):
    if spk not in spk2utt_dic:
        print("Cannot find {:s} in sp2utt".format(spk))
        exit(1)

    spk_utt_list = spk2utt_dic[spk]
    data_buf = [data_pool_dic[x] for x in spk_utt_list]
    
    return data_buf

def main(spk2utt_file, xvec_file, output_dir, output_name):
    
    # load spk2utt
    spk2utt = local_lib.load_utt(spk2utt_file)
    
    # load all the xvectors
    pool_xvectors = local_lib.load_scp(xvec_file)

    ave_data_dic = {}
    # compute average for each spk
    for spk in spk2utt.keys():
        # get data
        spk_data_buf = collect_data_spk(pool_xvectors, spk2utt, spk)
        
        if len(spk_data_buf) < 1:
            print("Cannot find data for {:s}".format(spk))
            exit(1)
        else:
            print("{:s} has {:d} data vectors".format(spk, len(spk_data_buf)))
    
        # get average
        ave_data = average_vectors(spk_data_buf)

        # save 
        ave_data_dic[spk] = ave_data
    
    # save
    output_file = os.path.join(output_dir, output_name)
    local_lib.write_scp(ave_data_dic, output_file)
    return

if __name__ == "__main__":

    # parse input
    args = sys.argv
    if len(args) != 4 and len(args) != 5:
        print(__doc__)
        exit(1)

    spk2utt_file = args[1]
    xvec_file = args[2]
    output_dir = args[3]
    if len(args) == 5:
        output_name = args[4]
    else:
        output_name = 'spk_xvector'

    main(spk2utt_file, xvec_file, output_dir, output_name)

