requirments:
data/$dataset_name/wav.scp - utt2spk - spk2utt - spk2gender


step1: bash selec_anon/compute_ori_spk_vector/00_extract_emb_fbank.sh --> selec_anon/output_ori_spk_vector
step2: bash selec_anon/compute_anon_spk_vector/01_run.sh  --> selec_anon/output_anon_spk_vector




