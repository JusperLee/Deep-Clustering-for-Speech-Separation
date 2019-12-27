import json
from pathlib import Path
from collections import OrderedDict
import torchaudio
import os
import torch

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


def read_wav(file_path, is_return_sr=False):
    '''
       file path: wav file path
       is_return_sr: if true, return sr number
    '''
    samp, sr = torchaudio.load(file_path)
    if is_return_sr:
        return samp.squeeze(), sr
    else:
        return samp.squeeze()


def write_wav(file_path, filename, samp, sr=8000):
    '''
       file_path: path of file
       filename: sound of Spectrogram
       sr: sample rate
    '''
    os.makedirs(file_path, exist_ok=True)
    filepath = os.path.join(file_path, filename)
    torchaudio.save(filepath, samp, sr)


def read_scp(scp_file):
    '''
      read the scp file
    '''
    files = open(scp_file,'r')
    lines = files.readlines()
    wave = {}
    for line in lines:
        line = line.split()
        if line[0] in wave.keys():
            raise ValueError
        wave[line[0]] = line[1]
    return wave


if __name__ == "__main__":
    print(read_scp('/home/likai/data1/create_scp/cv_mix.scp').keys())