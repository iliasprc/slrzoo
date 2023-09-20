import glob
import os
import random

import numpy as np
import torch
import torchvision
from PIL import Image
from omegaconf import OmegaConf

from base.base_loader import BaseDataset
from datasets.loader_utils import multi_label_to_index_out_of_vocabulary
from datasets.loader_utils import video_transforms, pad_video, sampling, VideoRandomResizedCrop
from utils.utils import load_csv_file

dataset_path = '/home/papastrat/Desktop/ilias/datasets/'
ssd_path = ''

phv1_path = 'datasets/phoenix2014-release/phoenix-2014-multisigner/features/fullFrame-210x260px/'
hands_path = 'datasets/phoenix2014-release/phoenix-2014-multisigner/features/h    '


class PHOENIX2014(BaseDataset):
    def __init__(self, config, args, mode, classes):
        super(PHOENIX2014, self).__init__(config, args, mode, classes)
        """
        Args:
            config ():
            mode ():

        """
        config = OmegaConf.load(os.path.join(args.cwd, "datasets/phoenix2014/dataset.yml"))['dataset']


        self.modality = config.modality
        self.mode = mode
        filepath = 'files/phoenix2014/' + mode + '_phoenixv1.csv'


        self.list_IDs, self.labels = load_csv_file(os.path.join(args.cwd, filepath))

        if (self.modality == 'full'):
            self.data_path = os.path.join(self.args.input_data, config.images_path,self.mode)
            #print(self.args.input_data, config.images_path,self.data_path)
            self.get = self.video_loader
        elif (self.modality == 'hand'):

            self.data_path = os.path.join(self.args.input_data, config.hand_image_path,self.mode)
        elif (self.modality == 'features'):
            self.data_path = os.path.join(self.args.input_data, config.features_path)
            self.get = self.feature_loader

        self.classes = classes
        self.num_classes = config.classes
        self.seq_length = config[mode]['seq_length']
        self.dim = config.dim
        self.normalize = config.normalize
        self.padding = config.padding
        self.augmentation = config[mode]['augmentation']
        self.img_type = config.img_type

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):

        return self.get(index)

    def feature_loader(self, index):
        y = multi_label_to_index_out_of_vocabulary(classes=self.classes, target_labels=self.labels[index])

        x = torch.tensor(np.load(os.path.join(self.data_path, self.mode, self.list_IDs[index].split('/')[0]) + '.npy'),
                         dtype=torch.float32).squeeze(0)

        return x, y

    def video_loader(self, index):

        y = multi_label_to_index_out_of_vocabulary(classes=self.classes, target_labels=self.labels[index])
        path = os.path.join(self.data_path, self.list_IDs[index])
        #print(self.data_path, self.list_IDs[index],self.args.input_data)
        images = sorted(glob.glob(os.path.join(path, '*' + self.img_type)))

        h_flip = False
        img_sequence = []

        before_len = len(images)

        if (self.augmentation):
            ## training set temporal  AUGMENTATION
            temporal_augmentation = int((np.random.randint(80, 100) / 100.0) * len(images))
            if (temporal_augmentation > 15):
                images = sorted(random.sample(images, k=temporal_augmentation))
            if (len(images) > self.seq_length):
                # random frame sampling
                images = sorted(random.sample(images, k=self.seq_length))

        else:
            # test uniform sampling
            if (len(images) > self.seq_length):
                images = sorted(sampling(images, self.seq_length))

        i = np.random.randint(0, 30)
        j = np.random.randint(0, 30)

        brightness = 1 + random.uniform(-0.1, +0.1)
        contrast = 1 + random.uniform(-0.1, +0.1)
        hue = random.uniform(0, 1) / 20.0

        r_resize = ((256, 256))

        gamma = 1. + random.uniform(-0.25, +0.25)
        grayscale = torchvision.transforms.Grayscale(num_output_channels=3) if (
                random.uniform(0., 1.) > 0.8) else torchvision.transforms.RandomGrayscale(
            p=0.0)  # torchvision.transforms.RandomGrayscale(p=0.2)

        t1 = VideoRandomResizedCrop(self.dim[0], scale=(0.9, 1.0), ratio=(0.8, 1.2))
        for img_path in images:

            frame = Image.open(img_path)
            frame.convert('RGB')

            if (self.augmentation):

                ## training set DATA AUGMENTATION

                frame = frame.resize(r_resize)

                img_tensor = video_transforms(img=frame, i=i, j=j, bright=brightness, cont=contrast, h=hue,
                                              dim=self.dim,
                                              resized_crop=t1,
                                              augmentation=self.augmentation, adjust_gamma=gamma, grayscale=grayscale,
                                              normalize=self.normalize)
                img_sequence.append(img_tensor)
            else:
                # TEST set  NO DATA AUGMENTATION
                # print('# TEST set  NO DATA AUGMENTATION')
                frame = frame.resize(self.dim)

                img_tensor = video_transforms(img=frame, i=i, j=j, bright=1, cont=1, h=0, dim=self.dim,
                                              augmentation=self.augmentation,
                                              normalize=self.normalize)
                img_sequence.append(img_tensor)
        pad_len = self.seq_length - len(images)

        x1 = torch.stack(img_sequence).float()

        if (self.padding):
            x1 = pad_video(x1, padding_size=pad_len, padding_type='zeros')
        elif (len(images) < 16):
            x1 = pad_video(x1, padding_size=16 - len(images), padding_type='zeros')
        # if self.augmentation =='train':
        #     x1 = video_tensor_shuffle(x1)

        return x1, y

# d = PHOENIX2014([], 'train')
