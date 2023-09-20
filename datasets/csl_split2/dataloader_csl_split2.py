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
from datasets.loader_utils import video_transforms, pad_video, sampling, VideoRandomResizedCrop, \
    multi_label_to_index, read_csl_paths
from omegaconf import OmegaConf






class CSL_SPLIT2(Dataset):
    def __init__(self, config, args, mode, classes):
        """
            (b) Split II - unseen sentence test:
            We choose 94 sentences (94 × 50 = 3700 videos) for training, and the remaining 6 sentences (6 × 50 = 300
            videos)
            for testing. The sentences in testing set are different from
            which in training set, while the vocabulary in testing set is
            a subset of vocabulary in training set.

        Args:
            config:
            args:
            mode:
            classes:
        """
        super().__init__(config, args, mode, classes)

        config = OmegaConf.load(os.path.join(args.cwd, 'datasets/csl_split1/dataset.yml'))
        self.modality = config.modality
        self.mode = mode
        self.dim = config.dim
        self.num_classes = config.classes
        self.seq_length = config[self.mode]['seq_length']
        self.augmentation = config[self.mode]['augmentation']
        self.normalize = config.normalize
        self.padding = config.padding
        filepath = './files/csl/csl_split2_unseen_sentence_' + self.mode + '.txt'
        dataset_path = config.dataset.images_path
        self.list_IDs, self.labels = read_csl_paths(os.path.join(args.cwd, filepath))

        print("{} examples {}".format(self.mode, len(self.list_IDs)))

        if (self.modality == 'full'):

            self.data_path = os.path.join(args.input_data, dataset_path)
            self.get = self.feature_loader
        elif (self.modality == 'features'):

            self.data_path = os.path.join(args.input_data, dataset_path)

            self.get = self.feature_loader

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):

        if self.modality == 'image':

            return self.video_loader(index)

        elif self.modality == 'feats':
            return self.feature_loader(index)

    def feature_loader(self, index):

        numpy_path = self.data_path + self.list_IDs[index].split('extracted')[-1] + '/feats.npy'

        y = multi_label_to_index(classes=self.classes, target_labels=self.labels[index])

        x = torch.FloatTensor(np.load(numpy_path)).squeeze(0)
        return x, y

    def video_loader(self, index):

        ID = self.list_IDs[index]
        y = multi_label_to_index(classes=self.classes, target_labels=self.labels[index])

        x = self.load_video_sequence_uniform_sampling(
            path=self.data_path + self.list_IDs[index].split('extracted')[-1],
            img_type='jpg')
        return x, y

    def load_video_sequence_uniform_sampling(self, path,
                                             img_type='png'):
        images = sorted(glob.glob(os.path.join(path, '*' + img_type)))

        img_sequence = []

        if (self.augmentation):

            temporal_augmentation = int((np.random.randint(80, 100) / 100.0) * len(images))
            if (temporal_augmentation > 15):
                images = sorted(sampling(images, temporal_augmentation))
            if (len(images) > self.seq_length):
                # random frame sampling
                images = sorted(sampling(images, self.seq_length))

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
                                              augmentation=self.augmentation,
                                              normalize=self.normalize)
                img_sequence.append(img_tensor)
            else:
                # TEST set  NO DATA AUGMENTATION
                frame = frame.resize(self.dim)

                img_tensor = video_transforms(img=frame, i=i, j=j, bright=0, cont=0, h=0, dim=self.dim,
                                              augmentation=False,
                                              normalize=self.normalize)
                img_sequence.append(img_tensor)
        pad_len = self.seq_length - len(images)

        X1 = torch.stack(img_sequence).float()

        if (self.padding):
            X1 = pad_video(X1, padding_size=pad_len, padding_type='zeros')
        elif (len(images) < 52):
            X1 = pad_video(X1, padding_size=52 - len(images), padding_type='zeros')

        return X1
