import glob
import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from datasets.loader_utils import pad_video, video_transforms, sampling, VideoRandomResizedCrop, read_signum_paths

dataset_path = '/home/papastrat/Desktop/ilias/datasets/SIGNUM'


class SIGNUM_ISO(Dataset):
    def __init__(self, args, mode, classes, dim=(224, 224), modality=None):
        """
        Args:
            mode : train or test and path prefix to read frames acordingly
            classes : list of classes
            channels: Number of channels of frames
            seq_length : Number of frames to be loaded in a sample
            dim: Dimensions of the frames
            normalize : normalize tensor with imagenet mean and std
            padding : padding of video to size seq_length



        """
        ## IN SIGNUM ISOLATED WE USE INDICES AS CLASSES
        filepath = './files/signum_isolated/signum_isolated_' + mode + '.txt'

        self.mode = mode

        print(self.mode, filepath)
        self.list_IDs, self.labels = read_signum_paths(filepath)
        self.classes = classes
        self.seq_length = args.seq_length
        self.mode = mode

        self.dim = dim

        self.normalize = args.normalize
        self.padding = args.padding

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        ID = self.list_IDs[index]
        # print(ID.split('SIGNUM')[-1],self.labels[index])

        y = torch.tensor(int(self.labels[index]) - 1, dtype=torch.long)

        x = self.load_video_sequence(path=(dataset_path + ID.split('SIGNUM')[-1]), time_steps=self.seq_length,
                                     dim=self.dim,
                                     augmentation=self.mode, padding=self.padding, normalize=self.normalize,
                                     img_type='jpg')

        return x, y

    def load_video_sequence(self, path, time_steps, dim=(224, 224), augmentation='test', padding=False, normalize=True,
                            img_type='png'):
        images = sorted(glob.glob(os.path.join(path, '*' + img_type)))
        images = images[7:73]

        h_flip = False
        img_sequence = []

        if (augmentation == 'train'):
            ## training set temporal  AUGMENTATION
            # temporal_augmentation = int((np.random.randint(80, 100) / 100.0) * len(images))
            # if (temporal_augmentation > 15):
            #     images = sorted(random.sample(images, k=temporal_augmentation))
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
            # print(np.array(frame).shape)
            frame1 = np.array(frame)[:, 42:300]
            frame = Image.fromarray(frame1)
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
        # print(len(img_sequence))
        X1 = torch.stack(img_sequence).float()

        if (padding):
            X1 = pad_video(X1, padding_size=pad_len, padding_type='zeros')

        return X1
