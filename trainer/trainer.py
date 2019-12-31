import time
import logging
from logger.set_logger import setup_logger
from model.loss import Loss
import torch
import sys
sys.path.append('../')


class Trainer(object):
    def __init__(self, train_dataloader, val_dataloader, optimizer, DPCL, opt):
        super(Trainer).__init__()
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.num_spks = opt['num_spks']
        if opt['is_gpu']:
            self.device = torch.device('cuda')
        self.dpcl = DPCL.to(self.device)
        self.optimizer = optimizer
        if opt['resume']['state']:
            self.resume_state = torch.load(
                opt['resume']['path'], map_location='cpu')
        self.print_freq = opt['logger']['print_freq']
        setup_logger(opt['logger']['name'], opt['logger']['path'],
                     screen=opt['logger']['screen'], tofile=opt['logger']['tofile'])
        self.logger = logging.getLogger(opt['logger']['name'])

    def train(self, epoch):
        self.dpcl.train()
        num_batchs = len(self.train_dataloader)
        total_loss = None
        num_index = 1
        start_time = time.time()
        for mix_wave, target_waves, non_slient in self.train_dataloader:
            mix_wave = mix_wave.to(self.device)
            target_waves = mix_wave.to(self.device)
            non_slient = non_slient.to(self.device)
            mix_embs = self.dpcl(mix_wave)
            l = Loss(mix_embs, target_waves, non_slient, self.num_spks)
            epoch_loss = l.loss()
            total_loss += epoch_loss
            self.optimizer.zero_grad()
            epoch_loss.backward()
            self.optimizer.step()
            if num_index % self.print_freq == 0:
                message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}>, loss:{:.3f}'.format(
                    epoch, num_index, self.optimizer.param_groups[0]['lr'], total_loss/num_index)
                self.logger.info(message)
        end_time = time.time()
        total_loss = total_loss/num_batchs
        message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}, loss:{:.3f}, Total time:{:.3f} min> '.format(
            epoch, num_batchs, self.optimizer.param_groups[0]['lr'], total_loss, (end_time-start_time)/60)
        self.logger.info(message)

    def validation(self, epoch):
        self.dpcl.eval()
        num_batchs = len(self.val_dataloader)
        num_index = 1
        total_loss = None
        start_time = time.time()
        with torch.no_grad():
            for mix_wave, target_waves, non_slient in self.val_dataloader:
                mix_embs = self.dpcl(mix_wave)
                l = Loss(mix_embs, target_waves, non_slient, self.num_spks)
                epoch_loss = l.loss()
                total_loss += epoch_loss
                if num_index % self.print_freq == 0:
                    message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}>, loss:{:.3f}'.format(
                        epoch, num_index, self.optimizer.param_groups[0]['lr'], total_loss/num_index)
                    self.logger.info(message)
        end_time = time.time()
        

