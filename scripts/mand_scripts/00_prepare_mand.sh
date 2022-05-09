#!/bin/sh
source env.sh

home=$PWD

#download pretrain models
if [ ! -e "pretrained_models/" ]; then
    if [ -f pretrained_models.tar.gz ];
    then
        rm pretrained_models.tar.gz
    fi
    echo -e "${RED}Downloading pre-trained model${NC}"

    #wget https://zenodo.org/record/6369772/files/pretrained_models.tar.gz
    cd /home/smg/miao/zenono/pretrained_models_sas_2022/pretrained_models_anon_xv/HuBERT_soft
    wget https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960.pt
    cd $home
fi



#download anonymized speaker vectors
if [ ! -e "data/aishell3" ]; then
    mkdir -p data
    cd data
    if [ -f aishell3.tar.gz ];
    then
        rm aishell3.tar.gz
    fi
    echo -e "${RED}Downloading alshell3 wavs and anonymized speaker vectors${NC}"

    wget https://zenodo.org/record/6371728/files/aishell3.tar.gz
    tar -xzvf aishell3.tar.gz
    cd ../
fi



home=$PWD

if [ ! -e "data/aishell3/wav16k_norm_enroll" ] || [ ! -e "data/aishell3/wav16k_norm_test" ]; then
   cd data/aishell3
   wget https://www.openslr.org/resources/93/data_aishell3.tgzv
   tar zxvf data_aishell3.tgz
fi

cd $PWD

## only use AISHELL3 test set
if [ -e "data/aishell3/train" ]; then
 rm -r data/aishell3/train
fi 

## downsample and normlization
SOURCE_DIR=data/aishell3/test/wav
TARGET_DIR=data/aishell3/test/wav16k_norm

SAMP=16000

##########
TMP=${TARGET_DIR}_TMP
mkdir -p ${TMP}
mkdir -p ${TARGET_DIR}

find $SOURCE_DIR -type f -name "*.wav" > file.lst
# step1. down-sampling

cat file.lst | parallel -j 20 sh scripts/sub_down_sample.sh {} ${TMP}/{/.}.wav ${SAMP}
#wait 

find ${TMP}  -type f -name "*.wav"  > file_tmp.lst
# step2.
cat file_tmp.lst | parallel -j 20 bash scripts/sub_sv56.sh {} ${TARGET_DIR}/{/.}.wav
wait 
rm -r $TMP
rm file.lst
rm file_tmp.lst


python scripts/mand_scripts/cp.py
