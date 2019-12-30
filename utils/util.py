import json
from pathlib import Path
from collections import OrderedDict
import librosa
import os
import sys
sys.path.append('../')
import torch
import numpy as np
from data_loader import AudioData
from tqdm import tqdm
import pickle
def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def read_wav(file_path, sr=8000, is_return_sr=False):
    '''
       file path: wav file path
       is_return_sr: if true, return sr number
    '''
    samp, sr = librosa.load(file_path, sr=sr)
    if is_return_sr:
        return samp, sr
    return samp


def write_wav(file_path, filename, samp, sr=8000):
    '''
       file_path: path of file
       filename: sound of Spectrogram
       sr: sample rate
    '''
    os.makedirs(file_path, exist_ok=True)
    filepath = os.path.join(file_path, filename)
    librosa.output.write_wav(filepath, samp, sr)


def read_scp(scp_file):
    '''
      read the scp file
    '''
    files = open(scp_file, 'r')
    lines = files.readlines()
    wave = {}
    for line in lines:
        line = line.split()
        if line[0] in wave.keys():
            raise ValueError
        wave[line[0]] = line[1]
    return wave


def compute_non_silent(samp, threshold=40, is_linear=True):
    '''
       samp: Spectrogram
       threshold: threshold(dB)
       is_linear: non-linear -> linear
    '''
    if is_linear:
        samp = np.exp(samp)
    samp_db = librosa.amplitude_to_db(samp,ref=1.0)
    max_mag = np.max(samp_db)
    threshold = librosa.db_to_amplitude(max_mag-samp_db)
    non_silent = np.array(samp > threshold, dtype=np.float32)
    return non_silent

def compute_cmvn(scp_file,save_file,**kwargs):
    '''
       Feature normalization
       scp_file: the file path of scp
       save_file: the cmvn result file .ark
       **kwargs: the configure setting of file

       return
            mean: [frequency-bins]
            var:  [frequency-bins]
    '''
    wave_reader = AudioData.AudioData(scp_file,**kwargs)
    frequency_bins = wave_reader[wave_reader.wave_keys[0]][0]
    all_wave = wave_reader[wave_reader.wave_keys[0]]
    index = 0
    for wave in tqdm(wave_reader):
        if index != 0:
            all_wave = np.row_stack((all_wave,wave))
        index+=1
    means = np.mean(all_wave,axis=0)
    stds = np.std(all_wave,axis=0)
    file_data = {'means':means,'stds':stds}
    pickle.dump(file_data,open(save_file,'wb'))
    print('means:',means)
    print('stds:',stds)
    
def apply_cmvn(samp,cmvn_dict):
    '''
      apply cmvn for Spectrogram
      samp: stft Spectrogram
      cmvn: the path of cmvn(python util.py)

      calculate: x = (x-mean)/std
    '''
    return (samp-cmvn_dict['means'])/cmvn_dict['stds']

if __name__ == "__main__":
    kwargs = {'window':'hann', 'nfft':256, 'window_length':256, 'hop_length':64, 'center':False, 'is_mag':True, 'is_log':True}
    compute_cmvn("/home/likai/data1/create_scp/cv_mix.scp",'./cmvn.ark',**kwargs)
    #file = pickle.load(open('cmvn.ark','rb'))
    #print(file)
    