import glob
import os

from PIL import Image
from torch.utils.data import Dataset
from base.base_loader import BaseDataset
SEED = 1234
import torch
import numpy as np
import random
from omegaconf import OmegaConf
from datasets.loader_utils import class2indextensor, pad_video, sampling, video_transforms, \
    VideoRandomResizedCrop, read_ph2014_isolated

dataset_path = 'data/phoenix2014_isolated_train/'
hand_path = 'data/phoenix2014_isolated_train/'

train_prefix = "train"
dev_prefix = "test"



class PHOENIX2014_ISO(BaseDataset):
    def __init__(self, config, args, mode, classes):
        super(PHOENIX2014_ISO, self).__init__(config, args, mode, classes)
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
        cwd_path = args.cwd
        config = OmegaConf.load(os.path.join(args.cwd, "datasets/phoenix2014isolated/dataset.yml"))['dataset']
        self.modality = config.modality
        self.mode = mode
        self.dim = config.dim
        self.num_classes = config.classes
        self.seq_length = config[self.mode]['seq_length']
        self.normalize = config.normalize
        self.padding = config.padding
        self.augmentation = config[self.mode]['augmentation']
        if (self.modality == 'hand'):
            train_filepath = os.path.join(args.cwd,'files/phoenix2014_isolated/hands_koller_10_1_iso_train.txt')
            dev_filepath = os.path.join(args.cwd,'files/phoenix2014_isolated/hands_koller_10_1_iso_test.txt')
            # dim = (112,112)
        else:
            train_filepath = os.path.join(args.cwd,"files/phoenix2014_isolated/train_split_80_20.txt")
            dev_filepath = os.path.join(args.cwd,"files/phoenix2014_isolated/test_split_80_20.txt")
        ### TO DO READ FILES IN HERE
        if (mode == train_prefix):

            list_IDs, labels = read_ph2014_isolated(train_filepath)
        elif (mode == dev_prefix):
            list_IDs, labels = read_ph2014_isolated(dev_filepath)
        print("Mode {} modality {} with {} examples".format(mode,self.modality,  len(list_IDs)))


        self.labels = labels
        self.list_IDs = list_IDs
       

        self.data_path = os.path.join(self.args.input_data, config.images_path)

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        ID = os.path.join(self.data_path, self.list_IDs[index])
        # print(ID)
        y = class2indextensor(classes=self.classes, target_label=self.labels[index])
        x = self.load_video_sequence_uniform_sampling(path=ID, time_steps=self.seq_length, dim=self.dim,
                                                      augmentation=self.mode, padding=self.padding,
                                                      normalize=self.normalize,
                                                      img_type='png')
        # print(x.shape)

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
        t1 = VideoRandomResizedCrop(dim[0], scale=(0.9, 1.0), ratio=(0.9, 1.1))
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
        if (len(img_sequence) < 1):
            img_sequence.append(torch.zeros((3, dim[0], dim[0])))
            print(path)
            X1 = torch.stack(img_sequence).float()
            print(X1.shape)
            X1 = pad_video(X1, padding_size=pad_len - 1, padding_type='zeros')
        # print(len(mages))
        elif (padding):
            X1 = torch.stack(img_sequence).float()
            # print(X1.shape)
            X1 = pad_video(X1, padding_size=pad_len, padding_type='zeros')

        return X1
