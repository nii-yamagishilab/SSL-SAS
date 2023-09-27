#!/bin/bash
#SBATCH --gres=gpu:tesla_a100:1
#SBATCH --time=3-00:00:00
#SBATCH -w gpuhost14
#source_dir=home/smg/miao/speech-re-facebook/unifi_fbank_ecapa_hifigan/scp/aishell/aishell_dev.lst
#outdir=aishell1_fbank_xvectors/anon


MAIN_ROOT=/home/smg/miao/anaconda3/envs/pytorch-1.8
if [ $(which python) != $MAIN_ROOT/bin/python ]; then source /home/smg/miao/anaconda3/bin/activate pytorch-1.8; fi


## extract libritts_train_other_500
source_dir=scp
#outdir=output_ori_spk_vector
outdir=test_24k
anon_pool="libritts_train_other_500"
python -m ipdb compute_ori_spk_vector/extract_emb_24k.py hparams/extract_ecapa_f_ecapa_vox.yaml $source_dir/$anon_pool/wav_24k.scp $outdir/$anon_pool fbank

exit 0

## extract vpc eval
for dset in libri_dev_{enrolls,trials_f,trials_m} \
              vctk_dev_{enrolls,trials_f_all,trials_m_all} \
              libri_test_{enrolls,trials_f,trials_m} \
              vctk_test_{enrolls,trials_f_all,trials_m_all}; do

python compute_ori_spk_vector/extract_emb.py hparams/extract_ecapa_f_ecapa_vox.yaml $source_dir/$dset/wav.scp $outdir/$dset fbank
done


<<!
## extract librispeech-360
source_dir=scp
outdir=output_ori_spk_vector
dset="libri_360"
python compute_ori_spk_vector/extract_emb.py hparams/extract_ecapa_f_ecapa_vox.yaml $source_dir/$dset/wav.scp $outdir/$dset fbank
!
