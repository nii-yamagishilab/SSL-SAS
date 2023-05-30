## Language independent SSL-based Speaker Anonymization system
This is an implementation of the papers:

(1) [Language-independent speaker anonymization approach using self-supervised pre-trained models](https://arxiv.org/abs/2202.13097) 

(2) [Analyzing Language-Independent Speaker Anonymization Framework under Unseen Conditions](https://arxiv.org/abs/2203.14834) 

(3) Language-independent speaker anonymization using orthogonal Householder neural network 

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
Mandarin models and speaker vectors are available for internal academic and research use only. If users would like to reproduce Mandarin anonymization experiments, please contact xiaoxiaomiao@nii.ac.jp. 

## Acknowledgments
This study is supported by JST CREST Grants (JPMJCR18A6 and JPMJCR20D3), MEXT KAKENHI Grants (21K17775, 21H04906, 21K11951, 18H04112), and the VoicePersonal project (ANR-18-JSTS-0001)

## License

The `adapted_from_facebookreaserch` subfolder has [Attribution-NonCommercial 4.0 International License](https://github.com/nii-yamagishilab/SSL-SAS/blob/main/adapted_from_facebookresearch/LICENSE). The `adapted_from_speechbrain` subfolder has [Apache License](https://github.com/nii-yamagishilab/SSL-SAS/blob/main/adapted_from_speechbrain/LICENSE). They were created by the [facebookreasearch](https://github.com/facebookresearch/speech-resynthesis/blob/main) and [speechbrain](https://github.com/speechbrain/speechbrain) orgnization, respectively. The `scripts` subfolder has the [MIT license](https://github.com/nii-yamagishilab/SSL-SAS/blob/main/scripts/LICENSE).

Because this source code was adapted from the facebookresearch and speechbrain, the whole project follows  
the [Attribution-NonCommercial 4.0 International License](https://github.com/nii-yamagishilab/SSL-SAS/blob/main/adapted_from_facebookresearch/LICENSE).

Copyright (c) 2022, Yamagishi Laboratory, National Institute of Informatics.
All rights reserved.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.



