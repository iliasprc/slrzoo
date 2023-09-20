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
import pathlib
from utils.utils import load_csv_file

dataset_path = '/home/papastrat/Desktop/ilias/datasets/'
ssd_path = ''
phv1_path = '/home/papastrat/Desktop/ilias/datasets/phoenix2014-release/phoenix-2014-multisigner/features/fullFrame-210x260px/'
hands_path = '/home/papastrat/Desktop/ilias/datasets/phoenix2014-release/phoenix-2014-multisigner/features/trackedRightHand-92x132px/'

# phv1_path='/media/papastrat/samsungssd/phoenix_version1/phoenix2014-release/phoenix-2014-multisigner/features/fullFrame-210x260px/'

from datasets.loader_utils import multi_label_to_index_out_of_vocabulary
from datasets.loader_utils import video_transforms, pad_video, sampling, VideoRandomResizedCrop


class Phoenix2014_2stream_Dataset(Dataset):
    def __init__(self, path_prefix, list_IDs, labels, classes, seq_length, dim=(224, 224), padding=False,
                 normalize=True):
        """
        Args:
            path_prefix : train or test and path prefix to read frames acordingly
            classes : list of classes
            channels: Number of channels of frames
            seq_length : Number of frames to be loaded in a sample
            dim: Dimensions of the frames
            normalize : normalize tensor with imagenet mean and std
            padding : padding of video to size seq_length

        """

        self.classes = classes
        self.seq_length = seq_length
        self.labels = labels
        self.list_IDs = list_IDs
        self.mode = path_prefix
        self.full_images_path = phv1_path + path_prefix + '/'
        self.hands_images_path = hands_path + path_prefix + '/'

        print("Modality used is ", self.images_path)
        self.mode = path_prefix
        filepath = './files/phoenix2014/' + path_prefix + '_phoenixv1.csv'
        print(self.mode, filepath)
        cwd_path = pathlib.Path.cwd()
        filepath = cwd_path.parent.joinpath(filepath)
        self.list_IDs, self.labels = load_csv_file(filepath)
        self.seq_length = seq_length

        self.padding = False
        self.normalize = True
        self.dim = dim

        self.normalize = normalize
        self.padding = padding

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        ID = self.list_IDs[index]

        y = multi_label_to_index_out_of_vocabulary(classes=self.classes, target_labels=self.labels[index])

        full_frames = self.load_video_sequence(path=os.path.join(self.full_images_path, ID), time_steps=self.seq_length,
                                               dim=self.dim,
                                               augmentation=self.mode, padding=self.padding, normalize=self.normalize,
                                               img_type='png')

        hands_frame = self.load_video_sequence(path=os.path.join(self.hands_images_path, ID),
                                               time_steps=self.seq_length,
                                               dim=self.dim,
                                               augmentation=self.mode, padding=self.padding, normalize=self.normalize,
                                               img_type='png')

        # full_frame = self.load_image_sequence(self.full_images_path + ID, self.seq_length)
        # hands_frame = self.load_image_sequence(self.hands_images_path + ID, self.seq_length)
        print(full_frames.size(), hands_frame.size())
        # print(self.full_images_path+ID,'\n',self.hands_images_path+ID)

        return full_frames, hands_frame, y

    def load_video_sequence(self, path, time_steps, dim=(224, 224), augmentation='test', padding=False, normalize=True,
                            img_type='png'):
        # print(os.path.join(path, '*' + img_type))
        images = sorted(glob.glob(os.path.join(path, '*' + img_type)))

        h_flip = False
        img_sequence = []
        # print(len(images))

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
            # print(frame.shape)

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

                img_tensor = video_transforms(img=frame, i=i, j=j, bright=1, cont=1, h=0, dim=dim, augmentation='test',
                                              normalize=normalize)
                img_sequence.append(img_tensor)
        pad_len = time_steps - len(images)

        X1 = torch.stack(img_sequence).float()

        if (padding):
            X1 = pad_video(X1, padding_size=pad_len, padding_type='zeros')
        elif (len(images) < 16):
            X1 = pad_video(X1, padding_size=16 - len(images), padding_type='zeros')
        return X1
