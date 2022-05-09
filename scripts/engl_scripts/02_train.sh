#!/bin/sh
source env.sh

home=$PWD
<<!
#download pretrain models
if [ ! -e "pretrained_models/" ]; then
    if [ -f pretrained_models.tar.gz ];
    then
        rm pretrained_models.tar.gz
    fi
    echo -e "${RED}Downloading pre-trained model${NC}"
    wget https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960.pt
    cd $home
fi


if [ ! -e "data/LibriTTS/" ]; then
   # wget https://www.openslr.org/resources/60/train-clean-100.tar.gz
    tar -xvzf train-clean-100.tar.gz
fi

downsamp_norm=True
if [ ${downsamp_norm}=='True' ]; then

    cd $PWD
    ## downsample and normlization
    SOURCE_DIR=data/LibriTTS/train-clean-100/
    TARGET_DIR=data/libritts_100_wav16k_norm

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
fi
!
### multi gpus setting ###
#export CUDA_VISIBLE_DEVICES="0,1,2,3"
#python3 -m torch.distributed.launch --nproc_per_node 4 train.py \
if [ -e "data/libritts_100_wav16k_norm/" ];then
    ### single gpu setting ###
    ### train libri HiFi-GAN
    python3 adapted_from_facebookresearch/train.py \
	    --checkpoint_path checkpoints/libri_tts_clean_100_fbank_xv_ssl_freeze \
	    --config configs/libri_tts_clean_100_fbank_xv_ssl_freeze.json
fi

