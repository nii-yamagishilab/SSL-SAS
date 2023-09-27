#!/bin/bash
#SBATCH --gres=gpu:tesla_a100:1
#SBATCH --time=3-00:00:00
#SBATCH -w gpuhost14
#source_dir=home/smg/miao/speech-re-facebook/unifi_fbank_ecapa_hifigan/scp/aishell/aishell_dev.lst
#outdir=aishell1_fbank_xvectors/anon


source ./env.sh

config=configs/extract_ecapa_f_ecapa_vox.yaml
echo ${config}
## extract libritts_train_other_500
source_dir=data
outdir=selec_anon/output_ori_spk_vector
anon_pool="libritts_train_other_500"
#python selec_anon/compute_ori_spk_vector/extract_emb.py ${config} $source_dir/$anon_pool/wav.scp $outdir/$anon_pool fbank


## extract vpc eval
for dset in libri_dev_{enrolls,trials_f,trials_m} \
              vctk_dev_{enrolls,trials_f_all,trials_m_all} \
              libri_test_{enrolls,trials_f,trials_m} \
              vctk_test_{enrolls,trials_f_all,trials_m_all}; do
python selec_anon/compute_ori_spk_vector/extract_emb.py ${config} $source_dir/$dset/wav.scp $outdir/$dset fbank
done


## extract librispeech-360
outdir=selec_anon/output_ori_spk_vector
dset="libri_360"
python selec_anon/compute_ori_spk_vector/extract_emb.py ${config} $source_dir/$dset/wav.scp $outdir/$dset fbank

