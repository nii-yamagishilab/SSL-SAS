step1: bash compute_ori_spk_vector/00_extract_emb_fbank.sh --> output_ori_spk_vector
step2: bash compute_anon_spk_vector/01_run.sh  --> output_anon_spk_vector



requirments:
scp/$dataset_name/wav.scp - utt2spk - spk2utt - spk2gender

