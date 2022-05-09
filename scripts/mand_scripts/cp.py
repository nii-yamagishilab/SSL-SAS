import sys,os,shutil

source_dir = 'data/aishell3/test/wav16k_norm'
target_test_dir = 'data/aishell3/wav16k_norm_test'
target_enroll_dir = 'data/aishell3/wav16k_norm_enroll'

if not os.path.exists(target_test_dir):
    os.makedirs(target_test_dir)

if not os.path.exists(target_enroll_dir):
    os.makedirs(target_enroll_dir)

for line in open('scp/aishell3/aishell3_test_sep.lst'):
    name = line.strip().split('/')[-1]
    shutil.copy(source_dir + '/' + name, target_test_dir + '/' + name)

for line in open('scp/aishell3/aishell3_enroll_sep.lst'):
    name = line.strip().split('/')[-1]
    shutil.copy(source_dir + '/' + name, target_enroll_dir + '/' + name)
