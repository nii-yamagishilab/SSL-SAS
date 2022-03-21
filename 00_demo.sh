#!/bin/bash
#SBATCH --gres=gpu:tesla_a100:1
#SBATCH --time=3-00:00:00
#SBATCH -w gpuhost12
#SBATCH --cpus-per-gpu=2
module load cuda11.1
source env.sh

#download pretrain models
if [ ! -e "pretrained_models/" ]; then
    if [ -f pretrained_models.tar.gz ];
    then
        rm pretrained_models.tar.gz
    fi
    echo -e "${RED}Downloading pre-trained model${NC}"

    wget --quiet https://zenodo.org/record/6369772/files/pretrained_models.tar.gz
    tar -xzvf pretrained_models.tar.gz
fi



#download aishell3 wavs and anonymized speaker vectors
if [ ! -e "data/aishell3" ]; then
    mkdir -p data
    cd data
    if [ -f aishell3.tar.gz ];
    then
        rm aishell3.tar.gz
    fi
    echo -e "${RED}Downloading alshell3 wavs and anonymized speaker vectors${NC}"

    wget --quiet https://zenodo.org/record/6371728/files/aishell3.tar.gz
    tar -xzvf aishell3.tar.gz
    cd ../
fi

# try pre-trained model
if [ -e "data/aishell3/" ];then
  echo -e "${RED}Try pre-trained model${NC}"
  #for vector_type in {libri,mls_10,mand_10}; do
    vector_type=libri #mls_10,mand_10,libri
    model_type=multilan_fbank_xv_ssl_freeze #multilan_fbank_xv_ssl_freeze,libri_tts_clean_100_fbank_xv_ssl_freeze
   #for model_type in {multilan_fbank_xv_ssl_freeze,libri_tts_clean_100_fbank_xv_ssl_freeze}; do
    for dset in {aishell3_enroll_sep,aishell3_test_sep}; do
	    python inference.py --input_test_file scp/aishell3/$dset.lst \
		    --xv_dir data/aishell3/$dset/${vector_type}_pseudo_xvectors \
		    --checkpoint_file pretrained_models/pretrain_hifigan/$model_type \
		    --output_dir pretrained_models/output/$model_type/${dset}_${vector_type}
    done
    echo -e "${RED}Please check generated waveforms from pre-trained model in ./pretrained_models/output"
    
    #compute CER
    listdir='scp_cer'
    mkdir -p $listdir
    for dest in {aishell3_test_sep,aishell3_enroll_sep}; do
	    wavdir=pretrained_models/output/$model_type/${dest}_${vector_type}
	    testlist=$listdir/${model_type}_${dest}_${vector_type}
	    echo $wavdir $testlist
	    echo find $wavdir -type f -name "*_gen.wav"  $listdir/${model_type}_${dest}_${vector_type}
	    find $wavdir -type f -name "*_gen.wav" > $listdir/${model_type}_${dest}_${vector_type}
	    python inference_for_cer.py $wavdir $testlist
    done
    echo anonymized CER
    cat results/hyp/$model_type\_aishell3_test_sep\_${vector_type} results/hyp/$model_type\_aishell3_enroll_sep\_${vector_type} > temp
    python compute_cer.py --mode present scp/transcript_aishell3_test_veri temp


    #compute EER
    echo OA
    python speaker_verification_cosine.py configs/verification_ecapa.yaml \
	    --enroll_dir=data/aishell3/wav16k_norm_enroll \
	    --test_dir=pretrained_models/output/$model_type/aishell3_test_sep_${vector_type}
    echo AA
    python speaker_verification_cosine.py configs/verification_ecapa.yaml \
	    --enroll_dir=pretrained_models/output/$model_type/aishell3_enroll_sep_${vector_type} \
	    --test_dir=pretrained_models/output/$model_type/aishell3_test_sep_${vector_type}
   #done
  #done
else
    echo "Cannot find data/aishell3/"
fi
exit 0

if [ ! -e "data/multi_language/" ];then
    cd data
    if [ -f multi_language.tar.gz ];
    then
        rm multi_language.tar.gz
    fi
    echo -e "${RED}Download multilingual training data${NC}"

    #wget 
    tar -xzvf multi_language.tar.gz
    cd ../
fi

### multi gpus setting ###
#export CUDA_VISIBLE_DEVICES="0,1,2,3"
#python3 -m torch.distributed.launch --nproc_per_node 4 train.py \
if [ -e "data/multi_language/" ];then
    ### single gpu setting ###
    ### train multilan HiFi-GAN
    python3 train.py \
	   --checkpoint_path checkpoints/multilan_fbank_xv_ssl_freeze \
	   --config configs/multilan_fbank_xv_ssl_freeze.json

    ### train libri HiFi-GAN
    #python3 train.py \
	    #--checkpoint_path checkpoints/libri_tts_clean_100_fbank_xv_ssl_freeze \
	    #--config configs/libri_tts_clean_100_fbank_xv_ssl_freeze.json
fi

