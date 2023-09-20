import glob
import os
import random

import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset

from datasets.loader_utils import multi_label_to_index_out_of_vocabulary
from datasets.loader_utils import video_transforms, pad_video, sampling, VideoRandomResizedCrop
from utils.utils import load_csv_file

dataset_path = '/home/papastrat/Desktop/ilias/datasets/'
ssd_path = ''
phv1_path = '/home/papastrat/Desktop/ilias/datasets/ph2014feats/'


class PHOENIX2014_FEATS(Dataset):
    def __init__(self, args, mode, classes, dim=(224, 224)):
        """
        Args:
            mode : train or test and path prefix to read frames acordingly
            classes : list of classes
            modality : full image or hands image
            channels: Number of channels of frames
            seq_length : Number of frames to be loaded in a sample
            dim: Dimensions of the frames
            normalize : normalize tensor with imagenet mean and std
            padding : padding of video to size seq_length

        """

        self.mode = mode
        filepath = '../files/phoenix2014/' + mode + '_phoenixv1.csv'
        print(self.mode, filepath)
        self.list_IDs, self.labels = load_csv_file(filepath)
        self.classes = classes
        self.seq_length = args.seq_length

        self.mode = mode

        self.feats_path = args.input_data + 'ph2014feats/' + mode + '/'

        if (mode != 'train'):
            self.mode = 'test'

        self.dim = dim

        self.normalize = args.normalize
        self.padding = args.padding
        print("{} {} samples ".format(len(self.list_IDs), self.mode))

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):

        ID = self.list_IDs[index]

        y = multi_label_to_index_out_of_vocabulary(classes=self.classes, target_labels=self.labels[index])

        x = torch.tensor(np.load(os.path.join(self.feats_path, ID.split('/')[0]) + '.npy'),
                         dtype=torch.float32).squeeze(0)


        return x, y


    def feature_loader(self,index):
        y = multi_label_to_index_out_of_vocabulary(classes=self.classes, target_labels=self.labels[index])

        x = torch.tensor(np.load(os.path.join(self.feats_path, self.list_IDs[index].split('/')[0]) + '.npy'),
                         dtype=torch.float32).squeeze(0)

        return x, y

    def load_video_sequence(self, path, time_steps, dim=(224, 224), augmentation='test', padding=False, normalize=True,
                            img_type='png'):
        # print(os.path.join(path, '*' + img_type))
        images = sorted(glob.glob(os.path.join(path, '*' + img_type)))

        h_flip = False
        img_sequence = []
        # print(len(images))
        before_len = len(images)

        if (augmentation == 'train'):
            ## training set temporal  AUGMENTATION
            temporal_augmentation = int((np.random.randint(80, 100) / 100.0) * len(images))
            if (temporal_augmentation > 15):
                images = sorted(random.sample(images, k=temporal_augmentation))
            if (len(images) > time_steps):
                # random frame sampling
                images = sorted(random.sample(images, k=time_steps))

        else:
            # test uniform sampling
            if (len(images) > time_steps):
                images = sorted(sampling(images, time_steps))
        # print(images)
        i = np.random.randint(0, 30)
        j = np.random.randint(0, 30)
        # print("images before and after ",before_len,len(images))
        brightness = 1 + random.uniform(-0.1, +0.1)
        contrast = 1 + random.uniform(-0.1, +0.1)
        hue = random.uniform(0, 1) / 20.0

        r_resize = ((256, 256))

        # brightness = 1
        # contrast = 1
        # hue = 0
        gamma = 1. + random.uniform(-0.25, +0.25)
        grayscale = torchvision.transforms.Grayscale(num_output_channels=3) if (
                random.uniform(0., 1.) > 0.8) else torchvision.transforms.RandomGrayscale(
            p=0.0)  # torchvision.transforms.RandomGrayscale(p=0.2)

        t1 = VideoRandomResizedCrop(dim[0], scale=(0.9, 1.0), ratio=(0.8, 1.2))
        for img_path in images:

            frame = Image.open(img_path)
            frame.convert('RGB')
            # print(frame.shape)

            if (augmentation == 'train'):

                ## training set DATA AUGMENTATION

                frame = frame.resize(r_resize)

                img_tensor = video_transforms(img=frame, i=i, j=j, bright=brightness, cont=contrast, h=hue, dim=dim,
                                              resized_crop=t1,
                                              augmentation='train', adjust_gamma=gamma, grayscale=grayscale,
                                              normalize=normalize)
                img_sequence.append(img_tensor)
            else:
                # TEST set  NO DATA AUGMENTATION
                # print('# TEST set  NO DATA AUGMENTATION')
                frame = frame.resize(dim)

                img_tensor = video_transforms(img=frame, i=i, j=j, bright=1, cont=1, h=0, dim=dim, augmentation='test',
                                              normalize=normalize)
                img_sequence.append(img_tensor)
        pad_len = time_steps - len(images)

        X1 = torch.stack(img_sequence).float()

        if (padding):
            X1 = pad_video(X1, padding_size=pad_len, padding_type='zeros')
        elif (len(images) < 16):
            X1 = pad_video(X1, padding_size=16 - len(images), padding_type='zeros')
        # if augmentation =='train':
        #     X1 = video_tensor_shuffle(X1)

        return X1
