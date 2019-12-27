import sys
sys.path.append('../')

import torch
import librosa
import numpy as np
from utils import util

class STFT(object):
    '''
       using the librosa implement of stft
       windows: a window function
       nfft: length of the windowed signal after padding with zeros.
       window_length: window() of length win_length 
       hop_length: number of audio samples between adjacent STFT columns.
    '''

    def __init__(self, window='hann', nfft=256, window_length=256, hop_length=64):
        self.window = window
        self.nfft = nfft
        self.window_length = window_length
        self.hop_length = hop_length

    def stft(self, samp, is_mag=False):
        # is_mag: Whether the output is an amplitude value
        samp = samp.cpu().numpy()
        stft_r = librosa.stft(samp, n_fft=self.nfft, hop_length=self.hop_length,
                              win_length=self.window_length, window=self.window)
        if is_mag:
            stft_r = np.abs(stft_r)
            return torch.from_numpy(stft_r)
        return stft_r

    def istft(self, stft_samp):
        output = librosa.istft(stft_samp, hop_length=self.hop_length,
                               win_length=self.window_length, window=self.window)
        output = torch.from_numpy(output)
        return output.unsqueeze(0)


if __name__ == "__main__":
    samp  = util.read_wav('../1.wav')
    stft_i = STFT()
    stft = stft_i.stft(samp,is_mag=False)
    print(stft.shape)
    output = stft_i.istft(stft)
    print(output.shape)
    util.write_wav('..','1_1.wav',output)