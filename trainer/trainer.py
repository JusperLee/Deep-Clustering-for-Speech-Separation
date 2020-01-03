import sys
sys.path.append('../')

import time
import logging
from logger.set_logger import setup_logger
from model.loss import Loss
import torch
import os
import matplotlib.pyplot as plt


class Trainer(object):
    def __init__(self, train_dataloader, val_dataloader, DPCL,  optimizer, opt):
        super(Trainer).__init__()
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.num_spks = opt['num_spks']
        self.cur_epoch = 0
        self.total_epoch = opt['train']['epoch']
        self.early_stop = opt['train']['early_stop']

        self.print_freq = opt['logger']['print_freq']
        #setup_logger(opt['logger']['name'], opt['logger']['path'],
        #             screen=opt['logger']['screen'], tofile=opt['logger']['tofile'])
        self.logger = logging.getLogger(opt['logger']['name'])
        self.checkpoint = opt['train']['path']
        self.name = opt['name']

        if opt['train']['is_gpu']:
            self.logger.info('Load Nvida GPU .....')
            self.device = torch.device('cuda:0')
            self.dpcl = DPCL.to(self.device)
        else:
            self.logger.info('Load CPU ...........')
            self.device = torch.device('cpu')
            self.dpcl = DPCL.to(self.device)
            
        if opt['resume']['state']:    
            ckp = torch.load(opt['resume']['path'],map_location='cpu')
            self.cur_epoch = ckp['epoch']
            self.logger.info("Resume from checkpoint {}: epoch {:d}".format(
                opt['resume']['path'], self.cur_epoch))
            self.dpcl = DPCL.load_state_dict(ckp['model_state_dict']).to(self.device)
            self.optimizer = optimizer.load_state_dict(ckp['optim_state_dict'])
        else:
            self.dpcl = DPCL.to(self.device)
            self.optimizer = optimizer
        
        if opt['optim']['clip_norm']:
            self.clip_norm = opt['optim']['clip_norm']
            self.logger.info("Gradient clipping by {}, default L2".format(self.clip_norm))
        else:
            self.clip_norm = 0
        

    def train(self, epoch):
        self.logger.info('Start training from epoch: {:d}, iter: {:d}'.format(epoch, 1))
        self.dpcl.train()
        num_batchs = len(self.train_dataloader)
        total_loss = 0.0
        num_index = 1
        start_time = time.time()
        for mix_wave, target_waves, non_slient in self.train_dataloader:
            mix_wave = mix_wave.to(self.device)
            target_waves = target_waves.to(self.device)
            non_slient = non_slient.to(self.device)
            mix_embs = self.dpcl(mix_wave)
            l = Loss(mix_embs, target_waves, non_slient, self.num_spks)
            epoch_loss = l.loss()
            total_loss += epoch_loss.item()
            self.optimizer.zero_grad()
            epoch_loss.backward()
            
            if self.clip_norm:
                torch.nn.utils.clip_grad_norm_(self.dpcl.parameters(),self.clip_norm)

            self.optimizer.step()
            if num_index % self.print_freq == 0:
                message = '<epoch:{:d}, iter:{:d}, lr:{:.3e}, loss:{:.3f}>'.format(
                    epoch, num_index, self.optimizer.param_groups[0]['lr'], total_loss/num_index)
                self.logger.info(message)
            num_index += 1
        end_time = time.time()
        total_loss = total_loss/num_batchs
        message = '<epoch:{:d}, iter:{:d}, lr:{:.3e}, loss:{:.3f}, Total time:{:.3f} min> '.format(
            epoch, num_batchs, self.optimizer.param_groups[0]['lr'], total_loss, (end_time-start_time)/60)
        self.logger.info(message)
        return total_loss

    def validation(self, epoch):
        self.logger.info('Start Validation from epoch: {:d}, iter: {:d}'.format(epoch, 1))
        self.dpcl.eval()
        num_batchs = len(self.val_dataloader)
        num_index = 1
        total_loss = 0.0
        start_time = time.time()
        with torch.no_grad():
            for mix_wave, target_waves, non_slient in self.val_dataloader:
                mix_wave = mix_wave.to(self.device)
                target_waves = target_waves.to(self.device)
                non_slient = non_slient.to(self.device)
                mix_embs = self.dpcl(mix_wave)
                l = Loss(mix_embs, target_waves, non_slient, self.num_spks)
                epoch_loss = l.loss()
                total_loss += epoch_loss.item()
                if num_index % self.print_freq == 0:
                    message = '<epoch:{:d}, iter:{:d}, lr:{:.3e}, loss:{:.3f}>'.format(
                        epoch, num_index, self.optimizer.param_groups[0]['lr'], total_loss/num_index)
                    self.logger.info(message)
                num_index += 1
        end_time = time.time()
        total_loss = total_loss/num_batchs
        message = '<epoch:{:d}, iter:{:d}, lr:{:.3e}, loss:{:.3f}, Total time:{:.3f} min> '.format(
            epoch, num_batchs, self.optimizer.param_groups[0]['lr'], total_loss, (end_time-start_time)/60)
        self.logger.info(message)
        return total_loss
    
    def run(self):
        train_loss = []
        val_loss = []
        with torch.cuda.device(0):
            self.save_checkpoint(self.cur_epoch,best=False)
            v_loss = self.validation(self.cur_epoch)
            best_loss = v_loss
            self.logger.info("Starting epoch from {:d}, loss = {:.4f}".format(
                self.cur_epoch, best_loss))
            no_improve = 0
            # starting training part
            while self.cur_epoch < self.total_epoch:
                self.cur_epoch += 1
                t_loss = self.train(self.cur_epoch)
                v_loss = self.validation(self.cur_epoch)

                train_loss.append(t_loss)
                val_loss.append(val_loss)

                if v_loss >= best_loss:
                    no_improve += 1
                    self.logger.info('No improvement, Best Loss: {:.4f}'.format(best_loss))
                else:
                    best_loss = val_loss
                    no_improve = 0
                    self.save_checkpoint(self.cur_epoch,best=True)
                    self.logger.info('Epoch: {:d}, Now Best Loss Change: {:.4f}'.format(self.cur_epoch,best_loss))
                
                if no_improve == self.early_stop:
                    self.logger.info(
                        "Stop training cause no impr for {:d} epochs".format(
                            no_improve))
                    break
            self.save_checkpoint(self.cur_epoch,best=False)
            self.logger.info("Training for {:d}/{:d} epoches done!".format(
                self.cur_epoch, self.total_epoch))
        

        # draw loss image
        plt.title("Loss of train and test")
        x = [i for i in range(self.cur_epoch)]
        plt.plot(x, train_loss, 'b-', label=u'train_loss',linewidth=0.8)
        plt.plot(x, val_loss, 'c-', label=u'val_loss',linewidth=0.8)
        plt.legend()
        #plt.xticks(l, lx)
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.savefig('conv_tasnet_loss.png')


    def save_checkpoint(self, epoch, best=True):
        '''
           save model
           best: the best model
        '''
        os.makedirs(os.path.join(self.checkpoint,self.name),exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.dpcl.state_dict(),
            'optim_state_dict': self.optimizer.state_dict()
        },
        os.path.join(self.checkpoint,self.name,'{0}.pt'.format('best' if best else 'last')))
    
    

