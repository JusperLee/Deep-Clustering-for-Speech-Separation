import sys
sys.path.append('../')
from utils import util
import torch



class Loss(object):
    def __init__(self, mix_wave, target_waves, non_slient, num_spks):
        super(Loss).__init__()
        self.mix_wave = mix_wave
        self.target_waves = target_waves
        self.non_slient = non_slient
        self.num_spks = num_spks

    def loss(self):
        '''
           mix_wave: B x TF x D
           target_waves: B x T x F
           non_slient: B x T x F 
        '''
        B, T, F = non_slient.shape

        # B x TF x spks
        target_embs = torch.zeros(
            self.target_waves.view(B, T*F, self.num_spks).shape)
        target_embs.scatter_(2, self.target_waves.view(B, T*F, 1), 1)

        # B x TF x 1
        self.non_slient = self.non_slient.view(B, T*F, 1)

        self.mix_wave = self.mix_wave * self.non_slient
        self.target_waves = self.target_waves * self.non_slient

        vt_v = torch.norm(torch.bmm(self.mix_wave.t(), self.mix_wave), p=2)**2
        vt_y = troch.norm(torch.bmm(self.mix_wave.t(),
                                    self.target_waves), p=2)**2
        yt_t = torch.norm(torch.bmm(self.target_waves.t(),
                                    self.target_waves), p=2)**2
        
        return (vt_v-2*vt_y+yt_t)/(B*T*F)
