#!/bin/bash


UTTDIR=data
SRCDIR=$PWD/selec_anon/output_ori_spk_vector/
POOL_XVEC_DIR=libritts_train_other_500
SCOREDIR=$PWD/selec_anon/output_anon_spk_vector
mkdir -p ${SCOREDIR}

SPKXVECNAME=spk_xvector
AFFINITYDIR=affinity
PSEUDODIR=pseudo_xvectors
PSEUDODIRBINDIR=xvectors

xvec_dim=192
rand_level="spk"
cross_gender="false"
distance="cosine"
proximity="farthest"
#proximity="oldest"
rand_seed="2020"

#rand_seed=0
# compute average of each xvector of each speaker
#  if necessary, concatenate the xvector_*.scp into xvector.scp
for dirname in `ls ${SRCDIR}`
do  
    TMPNAME=`echo ${dirname} | sed 's:xvectors_::g'`
    SPK2UTT=${UTTDIR}/${TMPNAME}/spk2utt
    XVEC_SCP=${SRCDIR}/${dirname}/xvector.scp

    OUTPUTDIR=${SCOREDIR}/${dirname}
    TMPSCP=${SCOREDIR}/${dirname}/xvector.scp
    mkdir -p ${OUTPUTDIR}
    
    if [ ! -e ${XVEC_SCP} ];
    then
	SUBFILES=`ls ${SRCDIR}/${dirname}/*.scp`
	echo "python concatenate_scp.py ${TMPSCP} ${SUBFILES}"
	python selec_anon/compute_anon_spk_vector/concatenate_scp.py ${TMPSCP} ${SUBFILES}
    else
	cp ${XVEC_SCP} ${TMPSCP}
    fi

    echo "python speaker_xvector.py ${SPK2UTT} ${TMPSCP} ${OUTPUTDIR} ${SPKXVECNAME}"
    python selec_anon/compute_anon_spk_vector/speaker_xvector.py ${SPK2UTT} ${TMPSCP} ${OUTPUTDIR} ${SPKXVECNAME}
done


# compute affinity
for dirname in `ls ${SRCDIR} | grep -v ${POOL_XVEC_DIR}`
do    
    TMP_SRC_PATH=${SCOREDIR}/${dirname}/${SPKXVECNAME}.scp
    TMP_POOL_PATH=${SCOREDIR}/${POOL_XVEC_DIR}/${SPKXVECNAME}.scp
    TMP_OUTPUT_DIR=${SCOREDIR}/${dirname}/${AFFINITYDIR}
    echo "python compute_spk_pool_cosine.py ${TMP_SRC_PATH} ${TMP_POOL_PATH} ${TMP_OUTPUT_DIR}"
    python selec_anon/compute_anon_spk_vector/compute_spk_pool_cosine.py ${TMP_SRC_PATH} ${TMP_POOL_PATH} ${TMP_OUTPUT_DIR}
done

# compute pseudo vectors
for dirname in `ls ${SRCDIR} | grep -v ${POOL_XVEC_DIR}`
do

    TMPNAME=`echo ${dirname} | sed 's:xvectors_::g'`
    TMP_SPK2UTT=${UTTDIR}/${TMPNAME}/spk2utt
    TMP_SPK2GENDER=${UTTDIR}/${TMPNAME}/spk2gender
    
    TMPNAME=`echo ${POOL_XVEC_DIR} | sed 's:xvectors_::g'`
    TMP_POOL_SPK2GENDER=${UTTDIR}/${TMPNAME}/spk2gender
    
    TMP_POOL_DATA_PATH=${SCOREDIR}/${POOL_XVEC_DIR}/${SPKXVECNAME}.scp
    TMP_AFFINITY_PATH=${SCOREDIR}/${dirname}/${AFFINITYDIR}
    TMP_OUTPUT_DIR=${SCOREDIR}/${dirname}/${PSEUDODIR}
    mkdir -p ${TMP_OUTPUT_DIR}

    echo "python gen_pseudo_xvecs.py ${TMP_SPK2UTT} ${TMP_SPK2GENDER} ${TMP_POOL_SPK2GENDER} \
	   ${TMP_AFFINITY_PATH} ${TMP_POOL_DATA_PATH} ${TMP_OUTPUT_DIR} ${PSEUDODIRBINDIR} \
	   ${rand_level} ${cross_gender} ${proximity} ${rand_seed} ${xvec_dim}"
    python selec_anon/compute_anon_spk_vector/gen_pseudo_xvecs.py ${TMP_SPK2UTT} ${TMP_SPK2GENDER} ${TMP_POOL_SPK2GENDER} \
	   ${TMP_AFFINITY_PATH} ${TMP_POOL_DATA_PATH} ${TMP_OUTPUT_DIR} ${PSEUDODIRBINDIR} \
	   ${rand_level} ${cross_gender} ${proximity} ${rand_seed} ${xvec_dim}
    
    #rand_seed=$((rand_seed+1))
done
