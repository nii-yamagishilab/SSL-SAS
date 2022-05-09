#!/bin/bash
# ==============================================================================
# Copyright (c) 2022, Yamagishi Laboratory, National Institute of Informatics
# Author: Xiaoxiao Miao (xiaoxiaomiao@nii.ac.jp)
# All rights reserved.
# ==============================================================================

source env.sh


# try pre-trained model
if [ -e "data/aishell3/" ];then
   echo -e "${RED}Try pre-trained model${NC}"
   for vector_type in {libri,mls_10,mand_10}; do
    #vector_type=libri #mls_10,mand_10,libri
    #model_type=multilan_fbank_xv_ssl_freeze #multilan_fbank_xv_ssl_freeze,libri_tts_clean_100_fbank_xv_ssl_freeze
   for model_type in {multilan_fbank_xv_ssl_freeze,libri_tts_clean_100_fbank_xv_ssl_freeze}; do
    
    #for dset in {aishell3_enroll_sep,aishell3_test_sep}; do
    #     python adapted_from_facebookresearch/inference.py --input_test_file scp/aishell3/$dset.lst \
#		    --xv_dir data/aishell3/$dset/${vector_type}_pseudo_xvectors \
#		    --checkpoint_file pretrained_models/pretrain_hifigan/$model_type \
#		    --output_dir pretrained_models/output/$model_type/${dset}_${vector_type}
 #   done
 #   echo -e "${RED}Please check generated waveforms from pre-trained model in ./pretrained_models/output"
    
    #compute CER
#    listdir='scp_cer'
#    mkdir -p $listdir
#    for dest in {aishell3_test_sep,aishell3_enroll_sep}; do
#	    wavdir=$(pwd)/pretrained_models/output/$model_type/${dest}_${vector_type}
#	    testlist=$listdir/${model_type}_${dest}_${vector_type}
#	    echo $wavdir $testlist
#	    echo find $wavdir -type f -name "*_gen.wav"  $listdir/${model_type}_${dest}_${vector_type}
#	    find $wavdir -type f -name "*_gen.wav" > $listdir/${model_type}_${dest}_${vector_type}
#	    python adapted_from_speechbrain/inference_for_cer.py $wavdir $testlist
#    done
    echo $model_type $vector_type anonymized CER
    cat results/hyp/$model_type\_aishell3_test_sep\_${vector_type} results/hyp/$model_type\_aishell3_enroll_sep\_${vector_type} > temp
    python adapted_from_speechbrain/compute_cer.py --mode present scp/transcript_aishell3_test_veri temp


    #compute EER
    echo OA
    python adapted_from_speechbrain/speaker_verification_cosine.py configs/verification_ecapa.yaml \
	    --enroll_dir=data/aishell3/wav16k_norm_enroll \
	    --test_dir=pretrained_models/output/$model_type/aishell3_test_sep_${vector_type}
    echo AA
    python adapted_from_speechbrain/speaker_verification_cosine.py configs/verification_ecapa.yaml \
	    --enroll_dir=pretrained_models/output/$model_type/aishell3_enroll_sep_${vector_type} \
	    --test_dir=pretrained_models/output/$model_type/aishell3_test_sep_${vector_type}
   done
  done
else
    echo "Cannot find data/aishell3/"
fi

if [ ! -e "data/multi_language/" ];then
    cd data
    
    if [ -f multi_language.tar.gz ];
    then
        rm multi_language.tar.gz
    fi
    echo -e "${RED}Download multilingual training data${NC}"


    for part in multi_language.tar.gz.parta{a,b,c,d,e}; do
	if [-f ${part}]; then
	   rm ${part}
	fi
	
	wget https://zenodo.org/record/6369670/files/${part}
	exit_code=$?
	# if exit code is not 0 (failed), then return it
	test $exit_code -eq 0 || exit $exit_code
    done	
    cat multi_language.tar.gz.parta* >multi_language.tar.gz

    tar -xzvf multi_language.tar.gz
    cd ../
fi

### multi gpus setting ###
#export CUDA_VISIBLE_DEVICES="0,1,2,3"
#python3 -m torch.distributed.launch --nproc_per_node 4 train.py \
if [ -e "data/multi_language/" ];then
    ### single gpu setting ###
    ### train multilan HiFi-GAN
    python3 adapted_from_facebookresearch/train.py \
	   --checkpoint_path checkpoints/multilan_fbank_xv_ssl_freeze \
	   --config configs/multilan_fbank_xv_ssl_freeze.json

    ### train libri HiFi-GAN
    #python3 train.py \
	    #--checkpoint_path checkpoints/libri_tts_clean_100_fbank_xv_ssl_freeze \
	    #--config configs/libri_tts_clean_100_fbank_xv_ssl_freeze.json
fi

