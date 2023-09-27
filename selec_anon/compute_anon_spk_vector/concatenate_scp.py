#!/usr/bin/env python
""" Concatenate multiple files into a single one

$: python concatenate_scp.py output_file input_file_1 input_file_2 ...

"""
import os
import sys
from os.path import join, isdir
import numpy as np

import local_lib

def main(file_list, output_file):
    
    dic_buf = {}
    for file_path in file_list:
        data_dic = local_lib.load_scp(file_path)
        dic_buf = {**dic_buf, **data_dic}
    print("Concatenate {:d} data items".format(len(dic_buf)))
    local_lib.write_scp(dic_buf, output_file)
    return

if __name__ == "__main__":

    # parse input
    args = sys.argv
    if len(args) < 3:
        print(__doc__)
        exit(1)

    output_file = args[1]
    input_file = args[2:]
    main(input_file, output_file)

