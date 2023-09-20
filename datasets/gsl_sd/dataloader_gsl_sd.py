import glob
import os
import random

import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf

from base.base_loader import BaseDataset
from datasets.loader_utils import multi_label_to_index
from datasets.loader_utils import pad_video, video_transforms, sampling, VideoRandomResizedCrop
from datasets.loader_utils import read_gsl_continuous, read_bounding_box, read_gsl_prev_sentence

root_path = 'GSL_continuous/'
train_prefix = "train"
dev_prefix = "dev"
test_prefix = "test"
train_filepath = "files/GSL_continuous/GSL_SD_train.csv"
dev_filepath = "files/GSL_continuous/GSL_SD_dev.csv"
test_filepath = "files/GSL_continuous/GSL_SD_test.csv"

feats_path = 'gsl_cont_features/'


class GSL_SD(BaseDataset):
    def __init__(self, config, args, mode, classes):
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
        super(GSL_SD, self).__init__(config, args, mode, classes)
        config = OmegaConf.load(os.path.join(args.cwd, "datasets/gsl_sd/dataset.yml"))['dataset']

        self.mode = mode
        self.dim = config.dim
        self.num_classes = config.classes
        self.seq_length = config[self.mode]['seq_length']
        self.normalize = config.normalize
        self.padding = config.padding
        self.augmentation = config[self.mode]['augmentation']
        self.return_context = self.args.return_context
        self.modality = config.modality

        if mode == train_prefix:
            self.list_IDs, self.list_glosses = read_gsl_continuous(os.path.join(args.cwd, train_filepath))

        elif mode == dev_prefix:
            self.list_IDs, self.list_glosses = read_gsl_continuous(os.path.join(args.cwd, dev_filepath))

        elif mode == test_prefix:
            self.list_IDs, self.list_glosses = read_gsl_continuous(os.path.join(args.cwd, test_filepath))

        print("{} {} instances {} modality ".format(len(self.list_IDs), mode, args.modality))

        self.bbox = read_bounding_box(os.path.join(args.cwd, 'files/GSL_continuous/bbox_for_gsl_continuous.txt'))

        self.context = read_gsl_prev_sentence(self.list_IDs)  # , self.list_glosses)

        if (self.modality == 'full'):
            self.data_path = os.path.join(self.args.input_data, config.images_path)
            self.get = self.video_loader
        elif (self.modality == 'features'):
            self.data_path = os.path.join(self.args.input_data, config.features_path)
            print()
            self.get = self.feature_loader

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):

        if self.modality == 'image':

            return self.video_loader(index)

        elif self.modality == 'feats':

            x, y, context = self.feature_loader(index)
            if self.return_context:
                return x, [y, context]
            return x, y

    def feature_loader(self, index):
        folder_path = os.path.join(self.root_path, self.list_IDs[index])
        # print(folder_path)

        y = multi_label_to_index(classes=self.classes, target_labels=self.list_glosses[index])
        # print('target ',self.list_glosses[index],' context ',self.context[index])
        if self.context[index] != 'None':

            c = multi_label_to_index(classes=self.classes, target_labels=self.context[index])
        else:
            c = torch.tensor([0], dtype=torch.int)

        if os.path.exists(folder_path + '.npy'):
            x = torch.FloatTensor(np.load(folder_path + '.npy')).squeeze(0)
        else:
            # print(folder_path, ' =empty')
            x = 0.00000000001 * torch.randn(25, 1024)
        return x, y, c

    def video_loader(self, index):

        x = self.load_video_sequence(path=self.list_IDs[index], time_steps=self.seq_length, dim=self.dim,
                                     augmentation=self.mode, padding=self.padding, normalize=self.normalize,
                                     img_type='jpg')
        y = multi_label_to_index(classes=self.classes, target_labels=self.list_glosses[index])

        return x, y

    def load_video_sequence(self, path, time_steps, dim=(224, 224), augmentation='test', padding=False, normalize=True,
                            img_type='png'):

        images = sorted(glob.glob(os.path.join(self.root_path, path, '*' + img_type)))

        h_flip = False
        img_sequence = []
        bbox = self.bbox.get(path)
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

        i = np.random.randint(0, 30)
        j = np.random.randint(0, 30)

        brightness = 1 + random.uniform(-0.2, +0.2)
        contrast = 1 + random.uniform(-0.2, +0.2)
        hue = random.uniform(0, 1) / 20.0

        r_resize = ((256, 256))

        # brightness = 1
        # contrast = 1
        # hue = 0
        crop_or_bbox = random.uniform(0, 1) > 0.5
        t1 = VideoRandomResizedCrop(dim[0], scale=(0.9, 1.0), ratio=(0.8, 1.2))
        for img_path in images:

            frame_o = Image.open(img_path)
            frame_o.convert('RGB')

            crop_size = 120
            ## CROP BOUNDING BOX

            frame1 = np.array(frame_o)
            if augmentation == 'test':
                if bbox != None:
                    frame1 = frame1[:, bbox['x1']:bbox['x2']]
                else:
                    frame1 = frame1[:, crop_size:648 - crop_size]
            else:

                if crop_or_bbox:
                    frame1 = frame1[:, crop_size:648 - crop_size]
                elif bbox != None:
                    frame1 = frame1[:, bbox['x1']:bbox['x2']]
                else:
                    frame1 = frame1[:, crop_size:648 - crop_size]
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

                img_tensor = video_transforms(img=frame, i=i, j=j, bright=1, cont=1, h=0, dim=dim, augmentation='test',
                                              normalize=normalize)
                img_sequence.append(img_tensor)
        pad_len = time_steps - len(images)

        x1 = torch.stack(img_sequence).float()

        if (padding):
            x1 = pad_video(x1, padding_size=pad_len, padding_type='zeros')
        if (len(images) < 20):
            x1 = pad_video(x1, padding_size=20 - len(images), padding_type='zeros')
        return x1
