

## Language independent SSL-based Speaker Anonymization system
This is an implementation of the papers:

(1) [Language-independent speaker anonymization approach using self-supervised pre-trained models](https://arxiv.org/abs/2202.13097) 

(2) [Analyzing Language-Independent Speaker Anonymization Framework under Unseen Conditions](https://arxiv.org/abs/2203.14834) 

The authors are Xiaoxiao Miao, Xin Wang, Erica Cooper, Junichi Yamagishi, Natalia Tomashenko.




Audio samples can be found here:  https://nii-yamagishilab.github.io/SAS-audio-samples/

Please cite these papers if you use this code.

## Dependencies
`git clone https://github.com/nii-yamagishilab/SSL-SAS.git`

`cd SSL-SAS`

`bash scripts/install.sh`

Make sure sox and parallel are installed. 

If not: 

`source env.sh`

`conda install -c conda-forge sox`

`conda install -c conda-forge parallel`

## 





## English anonymization

- Try pre-trained model

     1. Download English development and evaluation data provided by the [VoicePrivacy2020 Challenge](https://github.com/Voice-Privacy-Challenge/Voice-Privacy-Challenge-2020): [VCTK](https://datashare.ed.ac.uk/handle/10283/3443)-subsets (vctk_dev and vctk_test) and [LibriSpeech](http://www.openslr.org/12/)-subsets (libri_dev and libri_test). Just run `bash adapted_from_vpc/00_download_testdata.sh`. The user will be requested the password, please contact [VoicePrivacy2020 Challenge organizers](https://github.com/Voice-Privacy-Challenge/Voice-Privacy-Challenge-2020).
     2. Generate anonymized speech: `bash scripts/engl_scripts/01_demo.sh`.
     3. Following the [VoicePrivacy2020 Challenge](https://github.com/Voice-Privacy-Challenge/Voice-Privacy-Challenge-2020) to compute the performance.
 
- Train a HiFi-GAN using [LibriTTS-100h](https://www.openslr.org/60/) on your own: `bash scripts/engl_scripts/02_train.sh`

## Mandarin anonymization
Mandarin speaker vectors are available for internal academic and research use only. If users would like to reproduce Mandarin anonymization experiments, please contact ~~xiaoxiaomiao@nii.ac.jp~~ xiaoxiao.miao@singaporetech.edu.sg. 

## Acknowledgments
This study is supported by JST CREST Grants (JPMJCR18A6 and JPMJCR20D3), MEXT KAKENHI Grants (21K17775, 21H04906, 21K11951, 18H04112), and the VoicePersonal project (ANR-18-JSTS-0001)

## License

The `adapted_from_facebookreaserch` subfolder has [Attribution-NonCommercial 4.0 International License](https://github.com/nii-yamagishilab/SSL-SAS/blob/main/adapted_from_facebookresearch/LICENSE). The `adapted_from_speechbrain` subfolder has [Apache License](https://github.com/nii-yamagishilab/SSL-SAS/blob/main/adapted_from_speechbrain/LICENSE). They were created by the [facebookreasearch](https://github.com/facebookresearch/speech-resynthesis/blob/main) and [speechbrain](https://github.com/speechbrain/speechbrain) orgnization, respectively. The `scripts` subfolder has the [MIT license](https://github.com/nii-yamagishilab/SSL-SAS/blob/main/scripts/LICENSE).

Because this source code was adapted from the facebookresearch and speechbrain, the whole project follows  
the [Attribution-NonCommercial 4.0 International License](https://github.com/nii-yamagishilab/SSL-SAS/blob/main/adapted_from_facebookresearch/LICENSE).

Copyright (c) 2022, Yamagishi Laboratory, National Institute of Informatics.
All rights reserved.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

## Some potential questions you may have and how to solve them:
`File "/home/ubuntu/miao/SSL-SAS/venv/lib/python3.8/site-packages/speechbrain/utils/profiling.py", line 11, in <module>
    from torch.autograd.profiler_util import (  # pytorch v1.10.1
ModuleNotFoundError: No module named 'torch.autograd.profiler_util'`

`File "/home/ubuntu/miao/SSL-SAS/venv/lib/python3.8/site-packages/speechbrain/utils/profiling.py", line 527, in <module>
    a: EventList, b: EventList, filter_by: str = "count",
NameError: name 'EventList' is not defined `

Open */speechbrain/utils/profiling.py and comment out the function that causes the error.


## Instructions for aonymization for your own dataset

1. **Prepare the Kaldi Format Data Structure:**
   Create a data structure in Kaldi format, such as `test`, which includes the following files:
   - `test/wav.scp`
   - `test/spk2gender`
   - `test/spk2utt`
   - `test/utt2spk`

   For example:
   [data/libri_dev_enrolls](https://github.com/nii-yamagishilab/SSL-SAS/tree/main/data/libri_dev_enrolls)

2. **Generate Original Speaker Vectors:**
   Save the original speaker vectors to the directory `selec_anon/output_ori_spk_vector/`. Use the following script:
   [00_extract_emb_fbank.sh](https://github.com/nii-yamagishilab/SSL-SAS/blob/main/selec_anon/compute_ori_spk_vector/00_extract_emb_fbank.sh#L31)

3. **Generate Pseudo Speaker Vectors:**
   Use the script [01_run.sh](https://github.com/nii-yamagishilab/SSL-SAS/blob/main/selec_anon/compute_anon_spk_vector/01_run.sh) to generate pseudo speaker vectors. This script will read all datasets under `selec_anon/output_ori_spk_vector/` by default, treat `libritts500` as the external pool, and compute pseudo vectors for other datasets sequentially.

   Note the default configurations of the script, which you can modify as needed:
   - [`rand_level="spk"`](https://github.com/nii-yamagishilab/SSL-SAS/blob/main/selec_anon/compute_anon_spk_vector/01_run.sh#L16): If the utterances come from the same speaker, they will share the same pseudo vector. Set as`rand_level="utt"` where each utterance is anonymized individually without any association.
   - [`cross_gender="false"`](https://github.com/nii-yamagishilab/SSL-SAS/blob/main/selec_anon/compute_anon_spk_vector/01_run.sh#L17): The anonymized speech and original speech will be of the same gender. Set as `cross_gender="true"`: The anonymized speech and original speech will be of different genders.
   - The script calculates the similarity between the original speaker vector and the pool speaker vectors, then selects the `WORLDS=200` most dissimilar vectors of the same gender from the pool. From these, `REGION=100` vectors are randomly chosen and averaged to create the anonymized speaker vector. Modify [gen_pseudo_xvecs.py](https://github.com/nii-yamagishilab/SSL-SAS/blob/main/selec_anon/compute_anon_spk_vector/gen_pseudo_xvecs.py#L18-L19) as you needed.
  

4. **Modify Input List and `xv_dir` to Generate Anonymized Speech:**
   Update the input list and `xv_dir` to generate anonymized speech by modifying the following lines in the script:
   [01_demo.sh](https://github.com/nii-yamagishilab/SSL-SAS/blob/main/scripts/engl_scripts/01_demo.sh#L50-L51)

That's all and good luck!
