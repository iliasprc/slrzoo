# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 10:49:41 2019

@author: papastrat

"""

import argparse
import datetime
import json
import os
import pathlib
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from omegaconf import OmegaConf
from torch.optim.lr_scheduler import ReduceLROnPlateau

from datasets.loader_utils import select_isolated_dataset
from models import SLR_video_encoder,featur_encoder
from trainer import train, validate
from utils.model_utils import save_checkpoint_slr, select_optimizer, load_checkpoint
from utils.utils import txt_logger


def arguments():
    parser = argparse.ArgumentParser(description='SLR fully supervised')
    parser.add_argument('--input-data', type=str, default='/home/iliask/Desktop/ilias/datasets/',
                        help='path to datasets')
    parser.add_argument('--dataset', type=str, default='phoenix_iso', metavar='rc',
                        help='slr dataset phoenix_iso, phoenix_iso_I5, ms_asl , signum_isolated , csl_iso')
    parser.add_argument('--mode', type=str, default='isolated', metavar='rc',
                        help='isolated or continuous')
    parser.add_argument('--model', type=str, default='GoogLeNet_TConvsAtt', help='subunet or cui or i3d ')

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
    parser.add_argument('--log-interval', type=int, default=1000, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save', action='store_true', default=True,
                        help='For Saving the current Model')
    parser.add_argument('--resume', action='store_true', default=True,
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


def main():
    now = datetime.datetime.now()
    args = arguments()
    config = OmegaConf.load(os.path.join(args.cwd, 'tests/configs/isolated.yml'))['trainer']

    dt_string = now.strftime("%d_%m_%Y_%H.%M.%S")
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    print('pyTorch VERSION:', torch.__version__)
    print('CUDA VERSION')

    print('CUDNN VERSION:', torch.backends.cudnn.version())
    print('Number CUDA Devices:', torch.cuda.device_count())
    # dd/mm/YY H:M:S
    dt_string = now.strftime("%d_%m_%Y_%H.%M.%S")
    # print("date and time =", dt_string)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if (args.cuda and torch.cuda.is_available()):
        torch.cuda.manual_seed(args.seed)
    cudnn.benchmark = True
    cudnn.deterministic = True
    training_generator, val_generator, test_generator, classes = select_isolated_dataset(args)

    model = SLR_video_encoder(args, len(classes))
    #model = featur_encoder(args, len(classes),mode='isolated')
    use_cuda = torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda else "cpu")
    print('device ', device)
    resume_epoch = 1
    # summary(model,torch.randn(1,32,3,224,224))
    if (args.resume):
        model.last_linear = torch.nn.Linear(1024, 1234)
        pth_file, _ = load_checkpoint(
            '/home/iliask/PycharmProjects/SLRcheckpoints/model_cui/dataset_phoenix_2014/best_date_19_02_2020_21.41.25/best_wer.pth',
            model, strict=True, load_seperate_layers=True)

        # model.isolated_fc = torch.nn.Linear(1024, len(classes))
    if (args.cuda and use_cuda):
        model = model.cuda()

    optimizer, scheduler = select_optimizer(model, config, checkpoint=None)
    cpkt_fol_name = './checkpoints/model_' + args.model + '/dataset_' + args.dataset + '/date_' + dt_string
    best_val_loss = 10000

    print("Checkpoint Folder {} ".format(cpkt_fol_name))

    for epoch in range(resume_epoch, args.epochs):

        tr = train(args, model, device, training_generator, optimizer, epoch)
        print("!!!!!!!!   VALIDATION   !!!!!!!!!!!!!!!!!!")
        val_loss, ts = validate(args, model, device, val_generator, epoch)
        if test_generator:
            print("!!!!!!!!   TEST    !!!!!!!!!!!!!!!")
            validate(args, model, device, test_generator, epoch)

        ## to do scheduler
        scheduler.step(val_loss)
        if not os.path.exists(cpkt_fol_name):
            print("Checkpoint Directory does not exist! Making directory {}".format(cpkt_fol_name))
            os.makedirs(cpkt_fol_name)

        # save training and test measurements
        with open(cpkt_fol_name + '/training_arguments.txt', 'w') as f:
            json.dump(args.__dict__, f, indent=2)
        txt_logger(cpkt_fol_name + '/training.txt', tr)
        txt_logger(cpkt_fol_name + '/validation.txt', ts)

        for_checkpoint = {'epoch': epoch,
                          'model_dict': model.state_dict(),
                          'optimizer_dict': optimizer.state_dict(),
                          'validation_loss': str(val_loss)}
        is_best = val_loss < best_val_loss
        if (is_best):
            best_val_loss = val_loss
            save_checkpoint_slr(model, optimizer, epoch, val_loss, cpkt_fol_name, 'best',
                                save_seperate_layers=True)

        else:
            save_checkpoint_slr(model, optimizer, epoch, val_loss, cpkt_fol_name, 'last', save_seperate_layers=True)


if __name__ == '__main__':
    main()
