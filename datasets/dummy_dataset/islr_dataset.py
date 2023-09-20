import random

import torch
from torch.utils.data import Dataset

from base.base_loader import BaseDataset


class RandomISLRdataset(BaseDataset):
    def __init__(self, config, args, mode, classes):

        super(RandomISLRdataset, self).__init__(config, args, mode, classes)
        self.classes = classes
        self.num_classes = self.config.classes
        self.seq_length = self.config[mode]['seq_length']
        self.dim = self.config.dim
        self.normalize = self.config.normalize
        self.padding = self.config.padding
        self.augmentation = self.config[mode]['augmentation']
        self.img_type = self.config.img_type
        self.num_samples = 100

        self.video_size = (self.seq_length, 3, self.dim[0], self.dim[1])

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate a random image tensor
        image = torch.rand(*self.video_size)

        # Generate a random target label
        target = random.randint(0, self.num_classes - 1)

        return image, target
