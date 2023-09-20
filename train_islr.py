# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 10:49:41 2019

@author: papastrat

"""
import logging
import os
import sys

from trainer.trainer_islr import Trainer_ISLR

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import torch.backends.cudnn as cudnn

import argparse
import datetime
import os
import pathlib
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from omegaconf import OmegaConf

from datasets.loader_utils import select_isolated_dataset
from logger.timer import Timer
from models import SLR_video_encoder

from utils import make_dirs_if_not_present
from utils.model_utils import select_optimizer, save_checkpoint_islr


def arguments():
    parser = argparse.ArgumentParser(description='Isolated SLR')
    parser.add_argument('--input-data', type=str, default='/home/iliask/Desktop/ilias/datasets/',
                        help='path to datasets')
    parser.add_argument('--dataset', type=str, default='dummy', metavar='rc',
                        help='slr dataset phoenix_iso, phoenix_iso_I5, ms_asl , signum_isolated , csl_iso')
    parser.add_argument('--mode', type=str, default='isolated', metavar='rc',
                        help='isolated or continuous')
    parser.add_argument('--model', type=str, default='GoogLeNet_TConvs', help='subunet or cui or i3d ')

    parser.add_argument('--batch-size', type=int, default=4, metavar='N',
                        help='input batch size for training (default: 1)')

    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.0001)')

    parser.add_argument('--cuda', action='store_true', default=True,
                        help='disables CUDA training')

    parser.add_argument('--seed', type=int, default=1234, metavar='S',
                        help='random seed (default: 1234)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save', action='store_true', default=True,
                        help='For Saving the current Model')
    parser.add_argument('--resume', action='store_true', default=False,
                        help='For Saving the current Model')

    parser.add_argument('--optim', type=str, default='adam', metavar='optim number', help='optimizer sgd or adam')

    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--pretrained_cpkt', type=str,
                        default='/home/iliask/PycharmProjects/SLR_GAN/checkpoints/model_CLSR/dataset_GSL_SI'
                                '/date_07_07_2020_10.16.07/generator.pth',
                        help='fs checkpoint')
    args = parser.parse_args()
    args.cwd = os.path.join(pathlib.Path.cwd(), '')

    return args


args = arguments()
config = OmegaConf.load(os.path.join(args.cwd, 'configs/islr/isolated.yml'))['trainer']
now = datetime.datetime.now()
dt_string = now.strftime("%d_%m_%Y_%H.%M.%S")
checkpoint_dir = './checkpoints/model_' + args.model + '/dataset_' + args.dataset + '/date_' + dt_string

log_filename = "train_" + Timer().get_time() + ".log"

log_folder = os.path.join(checkpoint_dir, 'logs/')
make_dirs_if_not_present(log_folder)

log_filename = os.path.join(log_folder, log_filename)

logging.captureWarnings(True)

name = 'train_islr'

logger = logging.getLogger(name)

# Set level

logger.setLevel(getattr(logging, 'INFO'))

formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d-%H:%M:%S",
)

# Add handlers
file_hdl = logging.FileHandler(log_filename)
file_hdl.setFormatter(formatter)
logger.addHandler(file_hdl)
# logging.getLogger('py.warnings').addHandler(file_hdl)
cons_hdl = logging.StreamHandler(sys.stdout)
cons_hdl.setFormatter(formatter)
logger.addHandler(cons_hdl)


def main():
    best_acc1 = 0
    now = datetime.datetime.now()

    dt_string = now.strftime("%d_%m_%Y_%H.%M.%S")
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    logger.info(f'PyTorch VERSION:{torch.__version__}')
    logger.info(f'CUDA VERSION')

    logger.info(f'CUDNN VERSION: {torch.backends.cudnn.version()}')
    logger.info(f'Number CUDA Devices: {torch.cuda.device_count()}')
    # dd/mm/YY H:M:S
    dt_string = now.strftime("%d_%m_%Y_%H.%M.%S")
    # logger.info("date and time =", dt_string)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if (args.cuda and torch.cuda.is_available()):
        torch.cuda.manual_seed(args.seed)
    cudnn.benchmark = True
    cudnn.deterministic = True
    training_generator, validation_generator, test_generator, classes = select_isolated_dataset(config, args)

    model = SLR_video_encoder(config, args, len(classes))
    # model = featur_encoder(args, len(classes),mode='isolated')
    use_cuda = torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda else "cpu")

    if (args.cuda and use_cuda):
        model = model.cuda()
    args.start_epoch = 0
    optimizer, scheduler = select_optimizer(model, config, checkpoint=None)
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            elif torch.cuda.is_available():
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            logger.info("=> loaded checkpoint '{}' (epoch {})"
                        .format(args.resume, checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))

        # model.isolated_fc = torch.nn.Linear(1024, len(classes))


    logger.info(f'{model}')

    writer_path = os.path.join(args.cwd, 'runs/model_CSLR' + '/dataset_' + args.dataset + '/date_' + dt_string)

    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(writer_path)
    logger.info(f"CPKT DIR = {checkpoint_dir} ")
    logger.info(f"Summarywriter = {writer_path}")
    trainer = Trainer_ISLR(args, model=model, optimizer=optimizer, config=config, logger=logger,
                                  data_loader=training_generator, writer=writer,
                                  valid_data_loader=validation_generator, test_data_loader=test_generator,
                                  lr_scheduler=scheduler,
                                  checkpoint_dir=checkpoint_dir)

    trainer.train()


    # for epoch in range(args.start_epoch, args.epochs):
    #
    #     tr = train(args, model, device, training_generator, optimizer, epoch)
    #     logger.info("--------------------------  VALIDATION --------------------------")
    #     ts = validate(args, model, device, val_generator, epoch)
    #     val_loss = ts[1]
    #     if test_generator:
    #         logger.info("---------------------   TEST    --------------------------")
    #         validate(args, model, device, test_generator, epoch)
    #
    #     ## to do scheduler
    #     scheduler.step(val_loss)
    #
    #     is_best = ts[-1] > best_acc1
    #     best_acc1 = max(ts[-1], best_acc1)
    #
    #     if not os.path.exists(checkpoint_dir):
    #         logger.info("Checkpoint Directory does not exist! Making directory {}".format(checkpoint_dir))
    #         os.makedirs(checkpoint_dir)
    #
    #     if (is_best):
    #
    #         save_checkpoint_islr(model, optimizer, epoch, val_loss, checkpoint_dir, 'best',
    #                              save_seperate_layers=True)
    #
    #     else:
    #         save_checkpoint_islr(model, optimizer, epoch, val_loss, checkpoint_dir, 'last', save_seperate_layers=True)


if __name__ == '__main__':
    main()
