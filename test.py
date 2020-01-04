import torch
from config.option import parse
from sklearn.cluster import KMeans
from data_loader import AudioData

class Separation(object):
    '''
        test deep clutsering model
        dpcl: model
        scp_file: path of scp file
        opt: parse(yml)
    '''
    def __init__(self,dpcl,scp_file,opt):
        super(Separation).__init__()
        self.dpcl = dpcl.cuda()
        self.waves = AudioData.AudioData(scp_file,**opt['datasets']['audio_setting'])
        self.kmeans = KMeans(opt['num_spks'])
