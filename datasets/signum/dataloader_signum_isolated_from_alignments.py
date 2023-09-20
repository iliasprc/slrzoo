import glob
import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from datasets.loader_utils import pad_video, video_transforms, sampling, VideoRandomResizedCrop
from utils.utils import read_SIGNUM_CONTINUOUS_pkl

dataset_path = 'data/SIGNUM/'

train_prefix = "train"
dev_prefix = "test"

train_filepath = "./files/signum_alignments/signum_alignments_train.txt"
dev_filepath = "./files/signum_alignments/signum_alignments_test"


def load_signum_frames_per_gloss(path):
    data = open(path, 'r').read().splitlines()
    for i in data:
        path, frames = i.split(',')
        frame_indices = [int(s) for s in frames.split(' ')]
        # print(path,frame_indices)
    return data


align_path = '/home/papastrat/Desktop/ilias/signum_align1merged.txt'


class VideoDataset(Dataset):
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

            The signerdependent subset of SIGNUM corpus contains 603 German
            SL sentences for training and 177 for testing, each sentence is
            performed by a native signer three times. The training corpus
            contains 11874 glosses and 416620 frames in total, with 455
            different gestural categories

        """
        ## IN SIGNUM ISOLATED WE USE INDICES AS CLASSES

        ### TO DO SPLIT 603 for train and 177 for test
        self.mode = mode
        filepath = './files/Signum_continuous/database.pkl'
        _, _, _, id2w, classes, w2id = read_SIGNUM_CONTINUOUS_pkl(filepath, mode=mode)

        ### self.list_IDs = read signum align file
        if (mode == train_prefix):

            self.list_IDs = load_signum_frames_per_gloss(train_filepath)
        elif (mode == dev_prefix):
            self.list_IDs = load_signum_frames_per_gloss(dev_filepath)
        print("EXAMPLES ", len(self.list_IDs))
        print(self.mode, filepath)

        self.classes = classes
        self.seq_length = args.seq_length
        self.mode = mode

        self.dim = dim
        self.id2w = id2w
        self.w2id = w2id
        self.normalize = True
        self.padding = args.padding

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        ID = self.list_IDs[index]
        # print(ID.split('SIGNUM')[-1],self.labels[index])

        x, y = self.load_video_sequence(index=ID, time_steps=self.seq_length, dim=self.dim,
                                        augmentation=self.mode, padding=self.padding, normalize=self.normalize,
                                        img_type='jpg')

        return x, y

    def load_video_sequence(self, index, time_steps, dim=(224, 224), augmentation='test', padding=False, normalize=True,
                            img_type='png'):

        path_and_label, frames = index.split(',')
        path, label = path_and_label.split(' ')

        frame_indices = [int(s) for s in frames.split(' ')]
        # print(frame_indices)
        start_frame = frame_indices[0]
        end_frame = frame_indices[-1]
        images = sorted(glob.glob(os.path.join(dataset_path + path, '*' + img_type)))
        images = images[start_frame:end_frame]
        # print(path, label,len(frame_indices) )
        # print("LEN {} frames {}".format(len(images),len(frame_indices)))
        h_flip = False
        img_sequence = []

        if (augmentation == 'train'):
            ## training set temporal  AUGMENTATION
            # temporal_augmentation = int((np.random.randint(80, 100) / 100.0) * len(images))
            # if (temporal_augmentation > 15):
            #     images = sorted(random.sample(images, k=temporal_augmentation))
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
        y = torch.tensor(int(self.w2id[label]), dtype=torch.long)
        # print(len(X1))
        return X1, y
