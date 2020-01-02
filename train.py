import sys
sys.path.append('./')

from trainer import Trainer
import torch
import argparse
from config import option
import logging
from logger import set_logger
from model import model
from data_loader import Dataloader, AudioData



def make_dataloader(opt):
    # make train's dataloader
    train_mix_reader = AudioData(
        opt['datasets']['train']['dataroot_mix'], **opt['datasets']['audio_setting'])
    train_target_readers = [AudioData(opt['datasets']['train']['dataroot_targets'][0], **opt['datasets']['audio_setting']),
                            AudioData(opt['datasets']['train']['dataroot_targets'][1], **opt['datasets']['audio_setting'])]
    train_dataset = Dataloader.dataset(train_mix_reader, train_target_readers)
    train_dataloader = Dataloader.dataloader(
        train_dataset, **opt['datasets']['dataloader_setting'])

    # make validation dataloader
    val_mix_reader = AudioData(
        opt['datasets']['val']['dataroot_mix'], **opt['datasets']['audio_setting'])
    val_target_readers = [AudioData(opt['datasets']['val']['dataroot_targets'][0], **opt['datasets']['audio_setting']),
                          AudioData(opt['datasets']['val']['dataroot_targets'][1], **opt['datasets']['audio_setting'])]
    val_dataset = Dataloader.dataset(train_mix_reader, train_target_readers)
    val_dataloader = Dataloader.dataloader(
        train_dataset, **opt['datasets']['dataloader_setting'])

    return train_dataloader, val_dataloader


def make_optimizer(params, opt):
    optimizer = getattr(torch.optim, opt['optim']['name'])
    if opt['optim']['name'] == 'Adam':
        optimizer = optimizer(
            params, lr=opt['optim']['lr'], weight_decay=opt['optim']['weight_decay'])
    else:
        optimizer = optimizer(params, lr=opt['optim']['lr'], weight_decay=opt['optim']
                              ['weight_decay'], momentum=opt['optim']['momentum'])
    
    return optimizer

def train():
    parser = argparse.ArgumentParser(
        description='Parameters for training Deep Clustering')
    parser.add_argument('--opt', type=str, help='Path to option YAML file.')
    args = parser.parse_args()
    opt = option.parse(args.opt)
    set_logger.setup_logger(opt['logger']['name'], opt['logger']['path'],
                            screen=opt['logger']['screen'], tofile=opt['logger']['tofile'])
    logger = logging.getLogger(opt['logger']['name'])

    logger.info("Building the model of Deep Clustering")
    dpcl = model.DPCL(**opt['DPCL'])

    logger.info("Building the optimizer of Deep Clustering")
    optimizer = make_optimizer(dpcl.parameters(),opt)

    logger.info('Building the dataloader of Deep Clustering')
    train_dataloader, val_dataloader = make_dataloader(opt)

    logger.info('Building the Trainer of Deep Clustering')
    trainer = Trainer(train_dataloader, val_dataloader, dpcl,optimizer,opt)
    trainer.run()


if __name__ == "__main__":
    train()

