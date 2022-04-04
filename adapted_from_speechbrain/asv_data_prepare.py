# This source code was adapted from speechbrain by Xiaoxiao Miao (NII, Japan).

import os
import csv
import logging
import glob
import random
import shutil
import sys  # noqa F401
import numpy as np
import torch
import torchaudio
from tqdm.contrib import tqdm
from speechbrain.dataio.dataio import (
    load_pkl,
    save_pkl,
)

logger = logging.getLogger(__name__)
OPT_FILE = "opt_voxceleb_prepare.pkl"
TRAIN_CSV = "train.csv"
DEV_CSV = "dev.csv"
TEST_CSV = "test.csv"
ENROL_CSV = "enrol.csv"
SAMPLERATE = 16000


DEV_WAV = "vox1_dev_wav.zip"
TEST_WAV = "vox1_test_wav.zip"
META = "meta"


def prepare_data(
    save_folder,
    enroll_dir,
    test_dir,
    splits=["test"],
    amp_th=5e-04,
    verification_pairs_file=None,
    source=None,
    split_speaker=False,
    random_segment=False,
    skip_prep=False,
):
    """
    Prepares the csv files for the Voxceleb1 or Voxceleb2 datasets.
    Please follow the instructions in the README.md file for
    preparing Voxceleb2.

    Arguments
    ---------
    data_folder : str
        Path to the folder where the original VoxCeleb dataset is stored.
    save_folder : str
        The directory where to store the csv files.
    verification_pairs_file : str
        txt file containing the verification split.
    splits : list
        List of splits to prepare from ['train', 'dev']
    split_ratio : list
        List if int for train and validation splits
    seg_dur : int
        Segment duration of a chunk in seconds (e.g., 3.0 seconds).
    amp_th : float
        removes segments whose average amplitude is below the
        given threshold.
    source : str
        Path to the folder where the VoxCeleb dataset source is stored.
    split_speaker : bool
        Speaker-wise split
    random_segment : bool
        Train random segments
    skip_prep: Bool
        If True, skip preparation.

    Example
    -------
    >>> from recipes.VoxCeleb.voxceleb1_prepare import prepare_voxceleb
    >>> data_folder = 'data/VoxCeleb1/'
    >>> save_folder = 'VoxData/'
    >>> splits = ['train', 'dev']
    >>> split_ratio = [90, 10]
    >>> prepare_voxceleb(data_folder, save_folder, splits, split_ratio)
    """

    if skip_prep:
        return
    # Create configuration for easily skipping data_preparation stage
    conf = {
        "enroll_dir": enroll_dir,
        "test_dir": test_dir,
        "splits": splits,
        "save_folder": save_folder,
        "split_speaker": split_speaker,
    }

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Setting ouput files
    save_opt = os.path.join(save_folder, OPT_FILE)



    msg = "\tCreating csv file for the VoxCeleb Dataset.."
    logger.info(msg)

    if "test" in splits:
        prepare_csv_enrol_test(
            enroll_dir, test_dir, save_folder, verification_pairs_file
        )



def prepare_csv_enrol_test(enroll_dir, test_dir, save_folder, verification_pairs_file):
    """
    Creates the csv file for test data (useful for verification)

    Arguments
    ---------
    data_folder : str
        Path of the data folders
    save_folder : str
        The directory where to store the csv files.

    Returns
    -------
    None
    """

    # msg = '\t"Creating csv lists in  %s..."' % (csv_file)
    # logger.debug(msg)

    csv_output_head = [
        ["ID", "duration", "wav", "start", "stop", "spk_id"]
    ]  # noqa E231



    test_lst_file = verification_pairs_file

    enrol_ids, test_ids = [], []

    # Get unique ids (enrol and test utterances)
    for line in open(test_lst_file):
        e_id = line.split(" ")[1].rstrip().split('/')[-1].split(".")[0].strip()
        t_id = line.split(" ")[2].rstrip().split('/')[-1].split(".")[0].strip()
            

        enrol_ids.append(e_id)
        test_ids.append(t_id)

    enrol_ids = list(np.unique(np.array(enrol_ids)))
    test_ids = list(np.unique(np.array(test_ids)))

    # Prepare enrol csv
    logger.info("preparing enrol csv")
    enrol_csv = []
    for id in enrol_ids:
        if "output" in enroll_dir:
            wav = enroll_dir + '/' + id + "_gen.wav"
        else:
            wav = enroll_dir + '/' + id + ".wav"

        # Reading the signal (to retrieve duration in seconds)
        signal, fs = torchaudio.load(wav)
        signal = signal.squeeze(0)
        audio_duration = signal.shape[0] / SAMPLERATE
        start_sample = 0
        stop_sample = signal.shape[0]
        [spk_id, utt_id] = [id.split("-")[0], id]

        csv_line = [
            id,
            audio_duration,
            wav,
            start_sample,
            stop_sample,
            spk_id,
        ]

        enrol_csv.append(csv_line)

    csv_output = csv_output_head + enrol_csv
    csv_file = os.path.join(save_folder, ENROL_CSV)

    # Writing the csv lines
    with open(csv_file, mode="w") as csv_f:
        csv_writer = csv.writer(
            csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )
        for line in csv_output:
            csv_writer.writerow(line)

    # Prepare test csv
    logger.info("preparing test csv")
    test_csv = []
    for id in test_ids:
        if "output" in test_dir:
            wav = test_dir + '/' + id + "_gen.wav"
        else:
            wav = test_dir + '/' + id + ".wav"

        # Reading the signal (to retrieve duration in seconds)
        signal, fs = torchaudio.load(wav)
        signal = signal.squeeze(0)
        audio_duration = signal.shape[0] / SAMPLERATE
        start_sample = 0
        stop_sample = signal.shape[0]
        [spk_id, utt_id] = [id.split("/")[-1].split('-')[0],id.split("/")[-1]]

        csv_line = [
            id,
            audio_duration,
            wav,
            start_sample,
            stop_sample,
            spk_id,
        ]

        test_csv.append(csv_line)

    csv_output = csv_output_head + test_csv
    csv_file = os.path.join(save_folder, TEST_CSV)

    # Writing the csv lines
    with open(csv_file, mode="w") as csv_f:
        csv_writer = csv.writer(
            csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )
        for line in csv_output:
            csv_writer.writerow(line)
