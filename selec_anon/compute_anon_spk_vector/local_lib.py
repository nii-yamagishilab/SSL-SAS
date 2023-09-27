#!/usr/bin/env python
""" Compute average of x-vectors as speaker xvectors

$: python speaker_xvector.py spk2utt xvector.scp outputdir output_name_prefix

"""
import os
import sys
from os.path import join, isdir
import numpy as np
from kaldiio import WriteHelper, ReadHelper


def load_scp(scp_file):
    pool_xvectors = {}
    c = 0
    with ReadHelper('scp:' + scp_file) as reader:
        for key, xvec in reader:
            #print key, mat.shape
            pool_xvectors[key] = xvec
            c += 1
        print("Read ", c, "pool xvectors")
    return pool_xvectors

def write_scp(data_dic, output_file_path):
    output_file_path = os.path.splitext(output_file_path)[0]
    print("Writing to: " + output_file_path)
    ark_scp_output = 'ark,scp:{:s}.ark,{:s}.scp'.format(
        output_file_path, output_file_path)
    with WriteHelper(ark_scp_output) as writer:
        for spk, xvec in data_dic.items():
            writer(spk, xvec)
    return


def load_utt(spk2utt_file):
    spk2utt = {}
    with open(spk2utt_file) as f:
        for line in f.read().splitlines():
            sp = line.split()
            spk2utt[sp[0]] = sp[1:]
    return spk2utt


def load_gender(spk2gender_file):
    spk2gender = {}
    with open(spk2gender_file) as f:
        for line in f.read().splitlines():
            sp = line.split()
            spk2gender[sp[0]] = sp[1]
    return spk2gender

def f_write_raw_mat(data, filename, data_format='f4', end='l'):
    """flag = write_raw_mat(data, filename, data_format='f4', end='l')
    Write data to file on the file system as binary data

    input
    -----
      data:     np.array, data to be saved
      filename: str, path of the file to save the data
      data_format:   str, data_format for numpy
                 default: 'f4', float32
      end: str   little endian 'l' or big endian 'b'?
                 default: 'l'

    output   
    ------
      flag: bool, whether the writing is done or not
    """
    if not isinstance(data, np.ndarray):
        print("Error write_raw_mat: input should be np.array")
        return False
    f = open(filename,'wb')
    if len(data_format)>0:
        if end=='l':
            data_format = '<'+data_format
        elif end=='b':
            data_format = '>'+data_format
        else:
            data_format = '='+data_format
        datatype = np.dtype(data_format)
        temp_data = data.astype(datatype)
    else:
        temp_data = data
    temp_data.tofile(f,'')
    f.close()
    return True
