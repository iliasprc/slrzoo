import glob
import os
import random

import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset

from datasets.loader_utils import class2indextensor
from datasets.loader_utils import pad_video, video_transforms, sampling, VideoRandomResizedCrop, \
    video_tensor_shuffle, im_augmentation, read_ph2014_cui_isolated

dataset_path = '/home/andrster/Desktop/scripts/Datasets/phoenix2014-release/phoenix-2014-multisigner/features/fullFrame-210x260px/train/'
dataset_path = '/media/tomastheod/Ssd/phoenix_version1/phoenix2014-release/phoenix-2014-multisigner/features/fullFrame-210x260px/train/'
dataset_path = '/home/papastrat/Desktop/ilias/datasets/phoenix2014-release/phoenix-2014-multisigner/features/fullFrame-210x260px/train/'
hands_path = '/home/papastrat/Desktop/ilias/datasets/phoenix_version1/phoenix2014-release/phoenix-2014-multisigner/features/trackedRightHand-92x132px/train/'
hand = 'HAND'

train_prefix = "train"
dev_prefix = "test"
train_filepath = "files/phoenix2014_isolated/cui_train.txt"
dev_filepath = "files/phoenix2014_isolated/cui_test.txt"


# torch.manual_seed(SEED)
# np.random.seed(SEED)
# random.seed(SEED)
#
# torch.cuda.manual_seed(SEED)
phv1_path = '/home/iliask/Desktop/ilias/datasets/ph2014feats/'

class PH2014_ISO_CUI(Dataset):
    def __init__(self, args, mode, classes, dim=(224, 224), modality='full'):
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
        self.modality = modality
        if (modality == 'full'):
            self.images_path = os.path.join(args.input_data,'phoenix2014-release/phoenix-2014-multisigner/features/fullFrame-210x260px/train/')
            self.get = self.video_loader
        elif modality == 'features':
            self.features_path = phv1_path
            self.get = self.feature_loader
        ### TO DO READ FILES IN HERE
        if (mode == train_prefix):

            list_IDs, labels, starts, ends = read_ph2014_cui_isolated(os.path.join(args.cwd,train_filepath))
        elif (mode == dev_prefix):
            list_IDs, labels, starts, ends = read_ph2014_cui_isolated(os.path.join(args.cwd,dev_filepath))
        print("{} {} examples".format(mode, len(list_IDs)))
        self.classes = classes
        self.seq_length = 16
        self.labels = labels
        self.list_IDs = list_IDs
        self.mode = mode
        self.starting_frames = starts
        self.ending_frames = ends

        self.dim = dim
        self.normalize = True
        self.padding = True

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):

        # print(ID)

        y = class2indextensor(classes=self.classes, target_label=self.labels[index])
        if self.modality == 'full':
            ID = os.path.join(self.images_path, self.list_IDs[index])
            x = self.video_loader(path=ID, start=self.starting_frames[index],
                                         end=self.ending_frames[index], time_steps=self.seq_length,
                                         dim=self.dim,
                                         augmentation=self.mode, padding=self.padding,
                                         normalize=self.normalize,
                                         img_type='png')
        elif self.modality == 'features':
            x,y = self.feature_loader(index)

        return x, y
    def feature_loader(self, index):
        start = self.starting_frames[index]
        end = self.ending_frames[index]
        y = class2indextensor(classes=self.classes, target_label=self.labels[index])

        x = torch.tensor(np.load(os.path.join(self.features_path, self.mode, self.list_IDs[index].split('/')[0]) + '.npy'),
                         dtype=torch.float32).squeeze(0)[int(start):int(end),...]
        print(x.shape)
        return x, y
    def video_loader(self, path, start, end, time_steps=300, dim=(224, 224), augmentation='test', padding=True,
                            normalize=True,
                            img_type='png'):
        # print(os.path.join(path, '*' + img_type))
        images = sorted(glob.glob(os.path.join(path, '*' + img_type)))

        # select_frames
        # print(path,start,end)
        # print(start)
        if (start == end):
            start = int(start) - 1
        images = images[int(start): int(end)]
        # print(len(images))
        if (len(images) == 0):
            print("PROBLEM AT PATH {} {} {} ".format(path, start, end))
        # print(path)
        h_flip = False
        img_sequence = []
        # print(len(images))

        if (augmentation == 'train'):
            # ## training set temporal  AUGMENTATION
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

        brightness = 1 + random.uniform(-0.25, +0.25)
        contrast = 1 + random.uniform(-0.25, +0.25)
        hue = random.uniform(0, 1) / 10.0

        r_resize = ((256, 256))

        # brightness = 1
        # contrast = 1
        # hue = 0

        t1 = VideoRandomResizedCrop(dim[0], scale=(0.8, 1.251), ratio=(0.8, 1.251))
        grayscale = torchvision.transforms.Grayscale(num_output_channels=3) if (
                random.uniform(0., 1.) > 0.8) else torchvision.transforms.RandomGrayscale(
            p=0.0)  # torchvision.transforms.RandomGrayscale(p=0.2)
        # adjust_gamma = torchvisiontransforms.functional.adjust_gamma(img,gamma=1. + random.uniform(-0.2, +0.2))
        # print(grayscale)
        gamma = 1. + random.uniform(-0.25, +0.25)
        for img_path in images:

            frame = Image.open(img_path)
            frame.convert('RGB')

            if (augmentation == 'train'):

                ## training set DATA AUGMENTATION

                frame = frame.resize(r_resize)

                img_tensor = video_transforms(img=frame, i=i, j=j, bright=brightness, cont=contrast, h=hue, dim=dim,
                                              resized_crop=t1,
                                              augmentation='train',
                                              grayscale=grayscale,
                                              adjust_gamma=gamma,
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
        # if (len(images)<16):
        #     X1 = pad_video(X1, padding_size=16-len(images), padding_type='zeros')
        if (augmentation == 'train'):
            X1 = video_tensor_shuffle(X1)

        return X1

    def load_video_sequence_cui_aug(self, path, start, end, time_steps, dim=(224, 224), augmentation='test',
                                    padding=False, normalize=True,
                                    img_type='png'):
        # print(os.path.join(path, '*' + img_type))
        images = sorted(glob.glob(os.path.join(path, '*' + img_type)))

        # select_frames
        # print(path,start,end)
        # print(start)
        if (start == end):
            start = int(start) - 1
        images = images[int(start): int(end)]
        # print(len(images))
        if (len(images) == 0):
            print("PROBLEM AT PATH {} {} {} ".format(path, start, end))
        # print(path)

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
        gamma = 1. + random.uniform(-0.25, +0.25)
        grayscale = torchvision.transforms.Grayscale(num_output_channels=3) if (
                random.uniform(0., 1.) > 0.8) else torchvision.transforms.RandomGrayscale(
            p=0.0)  # torchvision.transforms.RandomGrayscale(p=0.2)

        t1 = VideoRandomResizedCrop(dim[0], scale=(0.9, 1.0), ratio=(0.8, 1.2))

        px = cv2.cvtColor(cv2.imread(images[0]), cv2.COLOR_BGR2RGB)
        px = px.reshape((-1, 3)) / 255.
        px -= px.mean(axis=0)
        weight, vec = np.linalg.eig(np.cov(px.T))

        for img_path in images:
            img_tensor = cv2.resize(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB), (224, 224))
            img_sequence.append(img_tensor)

        pad_len = time_steps - len(images)
        imgs = np.stack(img_sequence, axis=0)
        # print(imgs.shape)

        ims_src, imss_tensors = im_augmentation(imgs, weight, vec)
        # print(ims_src,np.array(ims_src, dtype=np.uint8))
        # img1 = t(img).permute(1,2,0).numpy()
        # import cv2
        # for i in range(ims_src.shape[0]):
        #
        #     cv2.imshow('dsd',cv2.cvtColor(ims_src[i],cv2.COLOR_RGB2BGR))
        #     cv2.waitKey(100)
        X1 = imss_tensors
        #
        if (padding):
            X1 = pad_video(X1, padding_size=pad_len, padding_type='zeros')
        elif (len(images) < 16):
            X1 = pad_video(X1, padding_size=16 - len(images), padding_type='zeros')

        return X1
