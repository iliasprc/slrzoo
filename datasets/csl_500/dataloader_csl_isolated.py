import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import glob
import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from datasets.loader_utils import video_transforms, pad_video, sampling, VideoRandomResizedCrop,read_csl_paths




class CSL_ISO(Dataset):
    def __init__(self, args, mode, classes, dim=(224, 224), modality=None):

        """
        Args:
            Split I - signer independent test:
            We use the videos performed by 40 signers for training, and
            the remaining videos of 10 signers for testing. The sentences of training and testing sets are the same, while the
            signers are different.

            (b) Split II - unseen sentence test:
            We choose 94 sentences (94 × 50 = 3700 videos) for training, and the remaining 6 sentences (6 × 50 = 300 videos)
            for testing. The sentences in testing set are different from
            which in training set, while the vocabulary in testing set is
            a subset of vocabulary in training set.



            path_prefix : train or test and path prefix to read frames acordingly
            classes : list of classes
            channels: Number of channels of frames
            seq_length : Number of frames to be loaded in a sample
            dim: Dimensions of the frames
            normalize : normalize tensor with imagenet mean and std
            padding : padding of video to size seq_length



        """

        self.mode = mode
        # self.list_IDs = filepath
        if (self.mode == 'train'):

            self.list_IDs, self.labels = read_csl_paths('./files/csl_isolated/dictionary_chineze_isolated_train.txt')
        elif (self.mode == 'test'):
            self.list_IDs, self.labels = read_csl_paths('./files/csl_isolated/dictionary_chineze_isolated_test.txt')
        # print(self.list_IDs,self.labels)
        print("{} examples {}".format(self.mode, len(self.list_IDs)))
        self.classes = classes
        self.seq_length = args.seq_length
        self.dim = dim
        self.normalize = args.normalize
        self.padding = args.padding

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        ID = self.list_IDs[index]

        # print(ID.split('/'))
        label = int(ID.split('/')[-3])
        path = ID.split('ChineseIsolated')
        # print(path,label)
        y = torch.tensor(label, dtype=torch.long)
        x = self.load_video_sequence_uniform_sampling(
            path='/home/papastrat/Desktop/ilias/datasets/ChineseIsolated/' + path[-1], time_steps=self.seq_length,
            dim=self.dim,
            augmentation=self.mode, padding=self.padding, normalize=self.normalize,
            img_type='jpg')

        return x, y

    def load_video_sequence_uniform_sampling(self, path, time_steps, dim=(224, 224), augmentation='test', padding=False,
                                             normalize=True,
                                             img_type='png'):
        images = sorted(glob.glob(os.path.join(path, '*' + img_type)))
        # normalize = True
        # augmentation = 'test'
        h_flip = False
        img_sequence = []
        # print(images)
        if (augmentation == 'train'):
            ## training set temporal  AUGMENTATION
            temporal_augmentation = int((np.random.randint(80, 100) / 100.0) * len(images))
            if (temporal_augmentation > 15):
                images = sorted(sampling(images, temporal_augmentation))
            if (len(images) > time_steps):
                # random frame sampling
                images = sorted(sampling(images, time_steps))

        else:
            # test uniform sampling
            if (len(images) > time_steps):
                images = sorted(sampling(images, time_steps))
        # print(images)
        i = np.random.randint(0, 30)
        j = np.random.randint(0, 30)

        brightness = 1 + random.uniform(-0.1, +0.1)
        contrast = 1 + random.uniform(-0.1, +0.1)
        hue = random.uniform(0, 1) / 20.0

        r_resize = ((256, 256))

        # brightness = 1
        # contrast = 1
        # hue = 0
        t1 = VideoRandomResizedCrop(dim[0], scale=(0.9, 1.0), ratio=(0.8, 1.2))
        for img_path in images:

            frame = Image.open(img_path)
            frame.convert('RGB')

            if (augmentation == 'train'):

                ## training set DATA AUGMENTATION

                frame = frame.resize(r_resize)

                img_tensor = video_transforms(img=frame, i=i, j=j, bright=brightness, cont=contrast, h=hue, dim=dim,
                                              resized_crop=t1,
                                              augmentation='train',
                                              normalize=normalize)
                img_sequence.append(img_tensor)
            else:
                # TEST set  NO DATA AUGMENTATION
                frame = frame.resize(dim)

                img_tensor = video_transforms(img=frame, i=i, j=j, bright=0, cont=0, h=0, dim=dim, augmentation='test',
                                              normalize=normalize)
                img_sequence.append(img_tensor)
        pad_len = time_steps - len(images)

        X1 = torch.stack(img_sequence).float()
        # print(len(images))
        if (padding):
            X1 = pad_video(X1, padding_size=pad_len, padding_type='zeros')
        elif (len(images) < 52):
            X1 = pad_video(X1, padding_size=52 - len(images), padding_type='zeros')

        return X1
