# This source code was adapted from speechbrain by Xiaoxiao Miao (NII, Japan).
import sys,os
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence
from speechbrain.pretrained import EncoderDecoderASR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
asr_model = (EncoderDecoderASR.from_hparams(source="speechbrain/asr-transformer-aishell", run_opts={"device":"cuda"},savedir="pretrained_models/asr-transformer-aishell"))


class MyDataset(Dataset):
  def __init__(self,wavdir,wavlist):
    f = open(wavlist, 'r')
    filenames = []
    for line in f:
      temp = line.strip().split('/')[-1]
      if '_gen.wav' in temp:
          filenames.append(temp.replace('_gen.wav', ''))
      elif '.wav' in temp:
          filenames.append(temp.replace('.wav', ''))
      #temp = line.strip().split('/')[-1:]
      #filenames.append("".join(temp))
    self.wavdir = wavdir
    self.wavnames = filenames

  def __getitem__(self, idx):
    wavname = self.wavnames[idx]
    wavpath = os.path.join(self.wavdir, wavname)
    if 'output' in wavpath:
        wav = torchaudio.load(wavpath + '_gen.wav')[0]
    else:
        wav = torchaudio.load(wavpath + '.wav')[0]
    wav_len = wav.shape[1]
    return wav.squeeze(), wav_len, wavname

  def __len__(self):
    return len(self.wavnames)

  def collate_fn(self, batch):  ## make them all the same length with zero padding
    wavs, lens, wavnames = zip(*batch)
    wavs = list(wavs)
    output_wavs = pad_sequence(wavs, batch_first=True, padding_value=0.0)
    lens = torch.Tensor(lens) / output_wavs.shape[1]
    return output_wavs, lens, wavnames

print('Loading data')
wavdir = sys.argv[1]
testlist = sys.argv[2]
outfile_name = testlist.split('/')[-1].split('.')[0]
print("writing hyp to %s"%outfile_name)

testset = MyDataset(wavdir, testlist)
testloader = DataLoader(testset, batch_size=32, shuffle=False, num_workers=1, collate_fn=testset.collate_fn)

print('Starting prediction')
if not os.path.isdir('results/hyp'):
    os.makedirs('results/hyp')

f = open("results/hyp/%s"%outfile_name,'w')
for i, data in enumerate(testloader, 0):
    batch,lens,filenames = data
    batch = batch.to(device)
    lens = lens.to(device)
    predicted_words, _ = asr_model.transcribe_batch(batch, lens)
    for i in range(len(batch)):
        #f.write('%s %s\n'%(filenames[i],predicted_words[i]))
        #print(filenames[i].split('/')[-1].split('.')[0],predicted_words[i])
        f.write('%s %s\n'%(filenames[i].split('/')[-1].split('.')[0],predicted_words[i]))
f.close()



