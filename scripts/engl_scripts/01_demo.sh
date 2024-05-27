#!/bin/bash
# ==============================================================================
# Copyright (c) 2022, Yamagishi Laboratory, National Institute of Informatics
# Author: Xiaoxiao Miao (xiaoxiaomiao@nii.ac.jp)
# All rights reserved.
# ==============================================================================

source env.sh


#download pretrain models
if [ ! -e "pretrained_models_anon_xv/" ]; then
    if [ -f pretrained_models_anon_xv.tar.gz ];
    then
        rm pretrained_models_anon_xv.tar.gz
    fi
    echo -e "${RED}Downloading pre-trained model${NC}"

    wget https://zenodo.org/record/6529898/files/pretrained_models_anon_xv.tar.gz
    tar -xzvf pretrained_models_anon_xv.tar.gz
    cd pretrained_models_anon_xv/
    wget https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960.pt
    cd $home
fi

#xv_flag=extract # extract or provide
xv_flag=provide
extract_config=configs/extract_ecapa_f_ecapa_vox.yaml


# try pre-trained model
if [ -e "data/libri_dev/" ];then
   echo -e "${RED}Try pre-trained model${NC}"
   #for model_type in {multilan_fbank_xv_ssl_freeze,libri_tts_clean_100_fbank_xv_ssl_freeze}; do
   model_type=libri_tts_clean_100_fbank_xv_ssl_freeze   
   if [[ ${xv_flag} == "extract" ]]; then
	 echo "extract original spk vectors for pool and testsets using selection-based anonymizer"
	 # compute ori speaker vector for libriTTS-500 (pool) and evaluation sets (libri+vctk)
	 bash selec_anon/compute_ori_spk_vector/00_extract_emb_fbank.sh ${extract_config}
	 # use selection-based anonymizer generate anonymized speaker vectors
	 # https://ieeexplore.ieee.org/abstract/document/9829284
	 bash selec_anon/compute_anon_spk_vector/01_run.sh
	 xv_dir=selec_anon/output_anon_spk_vector
   elif [[ ${xv_flag} == 'provide' ]]; then
         xv_dir=pretrained_models_anon_xv/anon_spk_vector/
   fi

   for dset in  libri_dev_{enrolls,trials_f,trials_m} \
		libri_test_{enrolls,trials_f,trials_m}; do
        python adapted_from_facebookresearch/inference.py --input_test_file scp/vpc/$dset.lst \
		    --xv_dir $xv_dir/$dset/pseudo_xvectors/xvectors \
		    --checkpoint_file pretrained_models_anon_xv/HiFi-GAN/$model_type \
		    --output_dir output/$model_type/${dset}
   done
   echo -e "${RED}Please check generated waveforms from pre-trained model in ./pretrained_models/output"
    
fi



