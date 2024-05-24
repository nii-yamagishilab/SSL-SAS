#!/usr/bin/python3
from kaldiio import WriteHelper
"""Recipe for training a speaker verification system based on cosine distance.
The cosine distance is computed on the top of pre-trained embeddings.
The pre-trained model is automatically downloaded from the web if not specified.
This recipe is designed to work on a single GPU.

To run this recipe, run the following command:
    >  python speaker_verification_cosine.py hyperparams/verification_ecapa_tdnn.yaml

Authors
    * Hwidong Na 2020
    * Mirco Ravanelli 2020
"""
import os
import sys
import torch
import logging
import torchaudio
import speechbrain as sb
from tqdm.contrib import tqdm
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.metric_stats import EER, minDCF
from speechbrain.utils.data_utils import download_file
from speechbrain.utils.distributed import run_on_main
import numpy as np
import torch.nn.functional as F
# Compute embeddings from the waveforms
def compute_embedding(wavs, wav_lens):
    """Compute speaker embeddings.

    Arguments
    ---------
    wavs : Torch.Tensor
        Tensor containing the speech waveform (batch, time).
        Make sure the sample rate is fs=16000 Hz.
    wav_lens: Torch.Tensor
        Tensor containing the relative length for each sentence
        in the length (e.g., [0.8 0.6 1.0])
    """

    with torch.no_grad():
        feats = params["compute_features"](wavs)
        feats = params["mean_var_norm"](feats, wav_lens)
        embeddings = params["embedding_model"](feats, wav_lens)
        #embeddings = params["mean_var_norm_emb"](
        #    embeddings, torch.ones(embeddings.shape[0]).to(embeddings.device)
        #)
    return embeddings.squeeze(1)

def compute_embedding_ssl(wavs, wav_lens):
    """Compute speaker embeddings.

    Arguments
    ---------
    wavs : Torch.Tensor
        Tensor containing the speech waveform (batch, time).
        Make sure the sample rate is fs=16000 Hz.
    wav_lens: Torch.Tensor
        Tensor containing the relative length for each sentence
        in the length (e.g., [0.8 0.6 1.0])
    """

    with torch.no_grad():
        feats = params["wav2vec2"].to(params["device"])(wavs)
        embeddings = params["embedding_model"].to(params["device"])(feats, wav_lens)
    return embeddings.squeeze(1)

if __name__ == "__main__":
    # Logger setup
    logger = logging.getLogger(__name__)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.dirname(current_dir))

    # Load hyperparameters file with command-line overrides
    params_file, run_opts, overrides = sb.core.parse_arguments(sys.argv[1:2])
    with open(params_file) as fin:
        params = load_hyperpyyaml(fin, overrides)


    # We download the pretrained LM from HuggingFace (or elsewhere depending on
    # the path given in the YAML file). The tokenizer is loaded at the same time.
    run_on_main(params["pretrainer"].collect_files)
    params["pretrainer"].load_collected(params["device"])
    params["embedding_model"].eval()
    params["embedding_model"].to(params["device"])

    wav_scp = sys.argv[2]
    outdir = sys.argv[3]
    feat_type = sys.argv[4]
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    with WriteHelper('ark,scp:%s/xvector.ark,%s/xvector.scp'%(outdir,outdir)) as writer:
        # Computing embeddings
        for wav_info in open(wav_scp):
            if '/.' in wav_info:
                print("skip error wav: %s"%wav_info)
                continue
            if ' ' in wav_info.strip():
                wav_name, wav_path = wav_info.strip().split(' ')
            elif 'vox' in wav_info.strip():
                wav_path = wav_info.strip()
                wav_name = "-".join(wav_info.strip().split('/')[-3:])
                wav_name = wav_name.replace('.wav','')

            else:
                wav_path = wav_info.strip()
                wav_name = wav_info.strip().split('/')[-1].split('.')[0]
                
            wav, fs = torchaudio.load(wav_path)
            wav  = torchaudio.transforms.Resample(fs, 16000)(wav) # Paper didn't use this

            # Manage single waveforms in input
            if len(wav.shape) == 1:
                wav = wav.unsqueeze(0)
            # Assign full length if wav_lens is not assigned

            wav_lens = torch.ones(wav.shape[0], device=params["device"])

            # Storing waveform in the specified device
            wav, wav_lens = wav.to(params["device"]), wav_lens.to(params["device"])
            wav = wav.float()
            if feat_type == 'fbank':
                print("Extract ori emb for %s"%wav_name)
                emb = compute_embedding(wav,wav_lens)
                #print(emb)
            elif feat_type == 'ssl':
                emb = compute_embedding_ssl(wav,wav_lens)

            writer(wav_name, emb.cpu().numpy())
