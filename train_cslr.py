import logging
import os
import sys

from logger.timer import Timer

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import argparse
import datetime
import random
import pathlib
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from datasets.loader_utils import select_continouous_dataset
from models import SLR_video_encoder
from trainer import Trainer_CSLR_method
from utils import select_optimizer, make_dirs_if_not_present
from omegaconf import OmegaConf


def arguments():
    parser = argparse.ArgumentParser(description='Continuous sign language recognition training')
    parser.add_argument('--input-data', type=str, default='/home/papastrat/Desktop/ilias/datasets/',
                        help='path to datasets')
    parser.add_argument('--dataset', type=str, default='dummy', metavar='rc',
                        help='slr dataset  greek_SI phoenixI5 phoenix2014feats  phoenix2014 phoenix2014T  GSL_SI  '
                             'GSL_SD   csl_split1 csl_split2 ')
    parser.add_argument('--modality', type=str, default='features')
    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 1)')

    parser.add_argument('--epochs', type=int, default=25, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--model', type=str, default='GoogLeNet_TConvs', help='subunet or cui or i3d ')

    parser.add_argument('--cuda', action='store_true', default=True,
                        help='disables CUDA training')
    parser.add_argument('--gpu', type=str, default='1')
    parser.add_argument('--seed', type=int, default=1234, metavar='S',
                        help='random seed (default: 1234)')

    parser.add_argument('--log-interval', type=int, default=1000, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save', action='store_true', default=True,
                        help='For Saving the current Model')
    parser.add_argument('--resume', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='For Saving the current Model')

    ## GENERATOR ARGUMENTS

    parser.add_argument('--ctc', type=str, default='normal',
                        help='normal for vanilla-CTC or focal or ent_ctc or custom or weighted or aggregation or '
                             'stim_ctc')

    args = parser.parse_args()
    args.cwd = os.path.join(pathlib.Path.cwd(), '')

    return args


now = datetime.datetime.now()
args = arguments()
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
if (args.cuda and torch.cuda.is_available()):
    torch.cuda.manual_seed(args.seed)
cudnn.benchmark = False
cudnn.deterministic = True
dt_string = now.strftime("%d_%m_%Y_%H.%M.%S")
checkpoint_dir = os.path.join(args.cwd,
                              'checkpoints/model_CLSR' + '/dataset_' + args.dataset + '/date_' + dt_string)
config = OmegaConf.load(os.path.join(args.cwd, 'configs/phoenix2014.yaml'))['trainer']

log_filename = "train_" + Timer().get_time() + ".log"

log_folder = os.path.join(checkpoint_dir, 'logs/')
make_dirs_if_not_present(log_folder)

log_filename = os.path.join(log_folder, log_filename)

logging.captureWarnings(True)

name = 'train_cslr'

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
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    logger.info(f'pyTorch VERSION:{torch.__version__}')
    logger.info(f'CUDA VERSION')

    logger.info(f'CUDNN VERSION: {torch.backends.cudnn.version()}')
    logger.info(f'Number CUDA Devices: {torch.cuda.device_count()}')

    training_generator, validation_generator, test_generator, classes, id2w, w2id = select_continouous_dataset(args)
    logger.info(f'Classes {len(classes)}')
    model = SLR_video_encoder(config, args, len(classes))

    optimizer, scheduler = select_optimizer(model, config, checkpoint=None)

    device1 = torch.device('cuda:0')
    if (args.cuda and torch.cuda.is_available()):
        model = model.to(device1)

    logger.info(f'{model}')

    writer_path = os.path.join(args.cwd, 'runs/model_CSLR' + '/dataset_' + args.dataset + '/date_' + dt_string)

    writer = SummaryWriter(writer_path)
    logger.info(f"CPKT DIR = {checkpoint_dir} ")
    logger.info(f"Summarywriter = {writer_path}")
    trainer = Trainer_CSLR_method(args, model=model, optimizer=optimizer, config=config, logger=logger,
                                  data_loader=training_generator, writer=writer,
                                  valid_data_loader=validation_generator, test_data_loader=test_generator,
                                  lr_scheduler=scheduler,
                                  checkpoint_dir=checkpoint_dir, id2w=id2w)

    trainer.train()


if __name__ == '__main__':
    main()
