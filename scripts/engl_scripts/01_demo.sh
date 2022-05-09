#!/bin/bash
# ==============================================================================
# Copyright (c) 2022, Yamagishi Laboratory, National Institute of Informatics
# Author: Xiaoxiao Miao (xiaoxiaomiao@nii.ac.jp)
# All rights reserved.
# ==============================================================================

source env.sh


#download pretrain models
if [ ! -e "pretrained_models/" ]; then
    if [ -f pretrained_models.tar.gz ];
    then
        rm pretrained_models.tar.gz
    fi
    echo -e "${RED}Downloading pre-trained model${NC}"

    #wget https://zenodo.org/record/6369772/files/pretrained_models.tar.gz
    #cd /home/smg/miao/zenono/pretrained_models_sas_2022/pretrained_models_anon_xv/HuBERT_soft
    wget https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960.pt
    cd $home
fi



# try pre-trained model
if [ -e "data/aishell3/" ];then
   echo -e "${RED}Try pre-trained model${NC}"
   #for model_type in {multilan_fbank_xv_ssl_freeze,libri_tts_clean_100_fbank_xv_ssl_freeze}; do
   model_type=libri_tts_clean_100_fbank_xv_ssl_freeze   
   for dset in  libri_dev_{enrolls,trials_f,trials_m} \
	        vctk_dev_{enrolls,trials_f_all,trials_m_all} \
		libri_test_{enrolls,trials_f,trials_m} \
		vctk_test_{enrolls,trials_f_all,trials_m_all}; do
        python adapted_from_facebookresearch/inference.py --input_test_file scp/vpc/$dset.lst \
		    --xv_dir data/vpc/$dset/pseudo_xvectors/xvectors \
		    --checkpoint_file pretrained_models/pretrain_hifigan/$model_type \
		    --output_dir pretrained_models/output/$model_type/${dset}
   done
   echo -e "${RED}Please check generated waveforms from pre-trained model in ./pretrained_models/output"
    
fi



