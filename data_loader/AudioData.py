import sys
sys.path.append('../')

import utils.util as ut
from utils.stft_istft import STFT
import torch



class AudioData(object):
    '''
        Loading wave file
        scp_file: the scp file path
        other kwargs is stft's kwargs
        is_mag: if True, abs(stft)
    '''
    def __init__(self, scp_file, window='hann', nfft=256, window_length=256, hop_length=64,is_mag=False):
        self.wave = ut.read_scp(scp_file)
        self.wave_keys = [key for key in self.wave.keys()]
        self.STFT = STFT(window=window, nfft=nfft,
                         window_length=window_length, hop_length=hop_length)
        self.is_mag = is_mag
    def __len__(self):
        return len(self.wave_keys)

    def stft(self, wave_path):
        samp = ut.read_wav(wave_path)
        return self.STFT.stft(samp,self.is_mag)
    
    def __iter__(self):
        for key in self.wave_keys:
            yield self.stft(self.wave[key])
    
    def __getitem__(self,key):
        if key not in self.wave_keys:
            raise ValueError
        return self.stft(self.wave[key])

if __name__ == "__main__":
    ad = AudioData("/home/likai/data1/create_scp/cv_mix.scp")
    b = ad.wave_keys
    print(type(b))