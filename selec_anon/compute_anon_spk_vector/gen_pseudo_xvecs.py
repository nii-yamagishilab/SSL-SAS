#!/usr/bin/env python
"""Compute pseudo xvectos

$: python gen_pseudo_xvecs.py 

"""
import os
import sys
from os.path import basename, join
import operator

import numpy as np
import random
from kaldiio import WriteHelper, ReadHelper
import pickle
import local_lib

REGION = 100
WORLD = 200

g_gender_rev = {'M': 'F', 'F': 'M'}

# Core logic of anonymization by randomization
def select_random_xvec(top500, pool_xvectors, xvec_dim):
    # number of random xvectors to select out of pool
    #random100mask = np.random.random_integers(0, 199, NR)
    choose_spk = []
    random100mask = random.sample(range(WORLD), REGION)
    pseudo_spk_list = [x for i, x in enumerate(top500) if i in
                           random100mask]
    pseudo_spk_matrix = np.zeros((REGION, xvec_dim), dtype='float64')
    for i, spk_aff in enumerate(pseudo_spk_list):
        pseudo_spk_matrix[i, :] = pool_xvectors[spk_aff[0]]
        choose_spk.append(spk_aff[0])
    # Take mean of 100 randomly selected xvectors
    pseudo_xvec = np.mean(pseudo_spk_matrix, axis=0)
    return pseudo_xvec, choose_spk


def get_top_K(spk, affinity_scores_dir, gender, pool_spk2gender, 
              cross_gender, proximity):

    # Filter the affinity pool by gender
    affinity_pool = {}

    # If we are doing cross-gender VC, reverse the gender else gender remains same
    if cross_gender:
        gender = g_gender_rev[gender]

    #print("Filtering pool for spk: "+spk)
    with open(join(affinity_scores_dir, 'affinity_'+spk)) as f:
        for line in f.read().splitlines():
            sp = line.split()
            pool_spk = sp[1]
            af_score = float(sp[2])
            if pool_spk2gender[pool_spk] == gender:
                affinity_pool[pool_spk] = af_score

    # Sort the filtered affinity pool by scores
    if proximity == "farthest":
        sorted_aff = sorted(affinity_pool.items(), key=operator.itemgetter(1))
    elif proximity == "nearest":
        sorted_aff = sorted(affinity_pool.items(), key=operator.itemgetter(1),
                           reverse=True)
    elif proximity== "oldest" or proximity== "youngest":
        with open("/home/smg/miao/speech-re-facebook/japa_company_2022/data/pool_1300/spk_age_pool.pkl",'rb') as tf:
            new_dic = pickle.load(tf)
        same_age_dic= {}       
        for line in new_dic:
            pool_spk = line
            age = new_dic[line]
            if pool_spk2gender[pool_spk] == gender:
                same_age_dic[pool_spk] = age
        if proximity== "oldest":
            sorted_aff = sorted(same_age_dic.items(), key=operator.itemgetter(1),
                           reverse=True)
        elif proximity== "youngest":
            sorted_aff = sorted(same_age_dic.items(), key=operator.itemgetter(1))
        #WORLD=100
    # Select WORLD least affinity speakers and then randomly select REGION out of
    # them
    top_spk = sorted_aff[:WORLD]
    return top_spk, gender

def write_pseudo_vector(pseudo_xvec_map, pseudo_gender_map, 
                        pseudo_xvecs_dir, pseudo_bin_dir):
    # Write features as ark,scp
    print("Writing pseud-speaker xvectors to: " + pseudo_xvecs_dir)
    ark_scp_output = 'ark,scp:{}/{}.ark,{}/{}.scp'.format(
        pseudo_xvecs_dir, 'pseudo_xvector',
        pseudo_xvecs_dir, 'pseudo_xvector')
    with WriteHelper(ark_scp_output) as writer:
        for uttid, xvec in pseudo_xvec_map.items():
            writer(uttid, xvec)

    print("Writing pseudo-speaker spk2gender.")
    with open(join(pseudo_xvecs_dir, 'spk2gender'), 'w') as f:
        spk2gen_arr = [spk+' '+gender for spk, gender in pseudo_gender_map.items()]
        sorted_spk2gen = sorted(spk2gen_arr)
        f.write('\n'.join(sorted_spk2gen) + '\n')

    # write to binary format
    output_dir = os.path.join(pseudo_xvecs_dir, pseudo_bin_dir)
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    for uttid, xvec in pseudo_xvec_map.items():
        output_path = os.path.join(output_dir, uttid) + '.xvector'
        local_lib.f_write_raw_mat(xvec, output_path)

    return

def main(args):
    
    ## args
    src_spk2utt_file = args[1]
    src_spk2gender_file = args[2]
    pool_spk2gender_file = args[3]

    affinity_scores_dir = args[4]
    pool_xvec_scp = args[5]
    output_pseudo_xvecs_dir = args[6]
    output_pseudo_xvecs_bin_dir = args[7]
    
    rand_level = args[8]
    cross_gender = args[9] == "true"
    proximity = args[10]
    rand_seed = args[11]
    xvec_dim = int(args[12])
    
    ## 
    random.seed(rand_seed)

    if cross_gender:
        print("Opposite gender speakers will be selected.")
    else:
        print("Same gender speakers will be selected.")

    print("Randomization level: " + rand_level)
    print("Proximity: " + proximity)
    
    
    ##
    src_spk2gender = local_lib.load_gender(src_spk2gender_file)
    src_spk2utt = local_lib.load_utt(src_spk2utt_file)
    pool_spk2gender = local_lib.load_gender(pool_spk2gender_file)
    
    # 
    pool_xvectors = local_lib.load_scp(pool_xvec_scp)
    
    
    pseudo_xvec_map = {}
    pseudo_gender_map = {}
    pseudo_spk_map = {}
    
    for spk, gender in src_spk2gender.items():
        gender = gender.upper()
        print(spk,gender)
        
        # get the top_K
        top_spk, gender = get_top_K(spk, affinity_scores_dir, gender,
                                    pool_spk2gender, cross_gender, proximity)
        pseudo_gender_map[spk] = gender
        
        # get the pseudo vector
        if rand_level == 'spk':
            # For rand_level = spk, one xvector is assigned to all the utterances
            # of a speaker
            pseudo_xvec, choose_spks = select_random_xvec(top_spk, pool_xvectors, xvec_dim)
            # Assign it to all utterances of the current speaker
            for uttid in src_spk2utt[spk]:
                pseudo_xvec_map[uttid] = pseudo_xvec
        elif rand_level == 'utt':
            # For rand_level = utt, random xvector is assigned to all the utterances
            # of a speaker
            for uttid in src_spk2utt[spk]:
                # Compute random vector for every utt
                pseudo_xvec, choose_spks = select_random_xvec(top_spk, pool_xvectors, xvec_dim)
                # Assign it to all utterances of the current speaker
                pseudo_xvec_map[uttid] = pseudo_xvec
        else:
            print("rand_level not supported! Errors will happen!")

        pseudo_spk_map[spk] = choose_spks
        #print(choose_spks)

    with open(join(output_pseudo_xvecs_dir, 'choose-spks'), 'w') as f:
        spk2spks_arr = [spk+' '+' '.join(spks) for spk, spks in pseudo_spk_map.items()]
        f.write('\n'.join(spk2spks_arr) + '\n')
    f.close()

    write_pseudo_vector(pseudo_xvec_map, pseudo_gender_map, 
                        output_pseudo_xvecs_dir, output_pseudo_xvecs_bin_dir)
    return

if __name__ == "__main__":
    args = sys.argv
    main(args)
