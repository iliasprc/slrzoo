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
from datasets.loader_utils import class2indextensor
from datasets.loader_utils import pad_video, video_transforms, sampling, VideoRandomResizedCrop, read_gsl_isolated, read_bounding_box

root_path = 'Greek_isolated/GSL_isol/'
train_prefix = "train"
dev_prefix = "test"
test_augmentation = 'augment'
train_filepath = "files/GSL_isolated/train_greek_iso.csv"
dev_filepath = "files/GSL_isolated/dev_greek_iso.csv"



class GSL_ISO(Dataset):
    def __init__(self, args, mode, classes, dim=(224, 224)):
        """

        Args:
            mode : train or test and path prefix to read frames acordingly
            classes : list of classes
            channels: Number of channels of frames
            dim: Dimensions of the frames
        """
        self.classes = classes
        print('Classes {}'.format(len(classes)))
        cwd_path = pathlib.Path.cwd()
        self.bbox = read_bounding_box(cwd_path.joinpath('files/GSL_isolated/bbox_for_gsl_isolated.txt'))
        # print(self.bbox)
        if mode == train_prefix:
            self.list_video_paths, self.list_glosses = read_gsl_isolated(cwd_path.joinpath(train_filepath))
            print("{} {} instances  ".format(len(self.list_video_paths), mode))
            self.mode = mode
        elif mode == dev_prefix:
            self.list_video_paths, self.list_glosses = read_gsl_isolated(cwd_path.joinpath(dev_filepath))
            # print(self.list_video_paths)
            print("{} {} instances  ".format(len(self.list_video_paths), mode))
            self.mode = mode
        elif mode == test_augmentation:
            self.list_video_paths, self.list_glosses = read_gsl_isolated(cwd_path.joinpath(dev_filepath))
            # print(self.list_video_paths)
            print("{} {} instances  ".format(len(self.list_video_paths), mode))
            self.mode = 'train'

        self.root_path = args.input_data + root_path
        self.seq_length = args.seq_length
        self.dim = dim
        self.normalize = args.normalize
        self.padding = args.padding

    def __len__(self):
        return len(self.list_video_paths)

    def __getitem__(self, index):

        # print(self.list_glosses[index])
        y = class2indextensor(classes=self.classes, target_label=self.list_glosses[index])
        # y = multi_label_to_index1(classes=self.classes, target_labels=self.list_glosses[index])
        # print(folder_path)
        x = self.load_video_sequence(index, time_steps=self.seq_length, dim=self.dim,
                                     augmentation=self.mode, padding=self.padding, normalize=self.normalize,
                                     img_type='jpg')

        return x, y

    def load_video_sequence(self, index, time_steps, dim=(224, 224), augmentation='test', padding=False, normalize=True,
                            img_type='png'):
        # print(os.path.join(path, '*' + img_type))
        path = os.path.join(self.root_path, self.list_video_paths[index])
        images = sorted(glob.glob(os.path.join(path, '*' + img_type)))
        # print(path)
        h_flip = False
        img_sequence = []
        # print(len(images))
        # print(self.bbox)
        bbox = self.bbox.get(self.list_video_paths[index])
        # print(self.list_video_paths[index],self.bbox.get(self.list_video_paths[index]))

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
        brightness = 1 + random.uniform(-0.2, +0.2)
        contrast = 1 + random.uniform(-0.2, +0.2)
        hue = random.uniform(0, 1) / 10.0
        r_resize = ((256, 256))
        crop_or_bbox = random.uniform(0, 1) > 0.5
        to_flip = random.uniform(0, 1) > 0.9
        grayscale = random.uniform(0, 1) > 0.9


        if (len(images) == 0):
            print('frames zero ', path)

        t1 = VideoRandomResizedCrop(dim[0], scale=(0.8, 2), ratio=(0.8, 1.2))
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
                                              normalize=normalize,to_flip=to_flip,grayscale=grayscale)
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
        X1 = X1.permute(1,0,2,3)
        return X1
