import sys,os
#/Database/LibriTTS/train-other-500/1006/135212/1006_135212_000001_000005.wav
fp = open('wav.scp','w')
fp_utt2spk = open('utt2spk', 'w')
spk2utt={}
for line in open('wav.lst'):
    token = line.strip().split('/')[-1].split('.')[0]
    fp.write('%s %s\n'%(token, line.strip()))
    spk = token.split('_')[0]
    fp_utt2spk.write('%s %s\n'%(token, token.split('_')[0]))
    if spk not in spk2utt:
        spk2utt[spk] = []
    spk2utt[spk].append(token)

fp.close()
fp_utt2spk.close()

fp_spk2gender = open('spk2gender', 'w')
spks = []
gender_map = {'female': 'F', 'male': 'M' }
for line in open('SPEAKERS.txt'):
    if '|' in line and 'train-other-500' in line:
        temp = line.strip().split('|')
        spk = temp[0].strip()
        gender = temp[1].strip()
        if spk not in spks:
            spks.append(spk)
            fp_spk2gender.write('%s %s\n'%(spk, gender))
fp_spk2gender.close()

fp_spk2utt = open('spk2utt','w')
for spk in spk2utt:
    line = []
    line.append(spk)
    for uu in spk2utt[spk]:
        line.append(uu)
    fp_spk2utt.write('%s\n'%' '.join(line))
fp_spk2utt.close()

    

