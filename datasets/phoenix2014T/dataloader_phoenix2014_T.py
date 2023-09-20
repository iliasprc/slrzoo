import glob
import os
import random

import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset

from datasets.loader_utils import multi_label_to_index_out_of_vocabulary
from datasets.loader_utils import video_transforms, pad_video, sampling, VideoRandomResizedCrop, load_phoenix_2014_T

dataset_path = 'data/'

dataset_path = 'data/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px/'
feats_path = '/media/papastrat/60E8EA1EE8E9F268/ph2014T/'

#
# torch.manual_seed(SEED)
# np.random.seed(SEED)
# random.seed(SEED)
#
# torch.cuda.manual_seed(SEED)


class PHOENIX2014T(Dataset):
    def __init__(self, args, mode, classes, dim=(224, 224), modality='full'):
        """
        Args:
            mode : train or test and path prefix to read frames acordingly
            classes : list of classes
            modality : full image or hands image
            channels: Number of channels of frames
            seq_length : Number of frames to be loaded in a sample
            dim: Dimensions of the frames
            normalize : normalize tensor with imagenet mean and std
            padding : padding of video to size seq_length

        """

        self.mode = mode
        filepath = './files/phoenix2014T/PHOENIX-2014-T.' + mode + '.corpus.csv'
        print(self.mode, filepath)
        self.list_IDs, self.labels = load_phoenix_2014_T(filepath)
        self.classes = classes
        self.seq_length = args.seq_length

        self.mode = mode
        if (modality == 'full'):
            self.path = dataset_path + mode + '/'
        elif (modality == 'feats'):
            self.path = feats_path + mode + '/'
        print("Modality used is ", self.path, 'examples ', len(self.list_IDs))
        if (mode != 'train'):
            self.mode = 'test'
            print("augmentation {}".format(self.mode))

        self.dim = dim

        self.normalize = args.normalize
        self.padding = args.padding

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):

        ID = self.list_IDs[index]
        # print(ID)
        y = multi_label_to_index_out_of_vocabulary(classes=self.classes, target_labels=self.labels[index])
        # print(y)
        # x = self.load_video_sequence(path=os.path.join(self.images_path, ID), time_steps=self.seq_length, dim=self.dim,
        #                              augmentation=self.mode, padding=self.padding, normalize=self.normalize,
        #                              img_type='png')
        x = torch.tensor(np.load(os.path.join(self.path, ID.split('/')[0]) + '.npy'),
                         dtype=torch.float32).squeeze(0)
        return x, y

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
        gamma = 1. + random.uniform(-0.25, +0.25)
        grayscale = torchvision.transforms.Grayscale(num_output_channels=3) if (
                random.uniform(0., 1.) > 0.8) else torchvision.transforms.RandomGrayscale(
            p=0.0)  # torchvision.transforms.RandomGrayscale(p=0.2)

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
                                              augmentation='train', adjust_gamma=gamma, grayscale=grayscale,
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
        # if augmentation =='train':
        #     X1 = video_tensor_shuffle(X1)

        return X1

###### CV2 augmentations
# def load_video_sequence_cui(self, path, time_steps, dim=(224, 224), augmentation='test', padding=False,
#                             normalize=True,
#                             img_type='png'):
#     # print(os.path.join(path, '*' + img_type))
#     # print(path)
#     images = sorted(glob.glob(os.path.join(path, '*' + img_type)))
#
#     h_flip = False
#     img_sequence = []
#     # print(len(images))
#
#     if (augmentation == 'train'):
#         ## training set temporal  AUGMENTATION
#         temporal_augmentation = int((np.random.randint(80, 100) / 100.0) * len(images))
#         if (temporal_augmentation > 15):
#             images = sorted(random.sample(images, k=temporal_augmentation))
#         if (len(images) > time_steps):
#             # random frame sampling
#             images = sorted(random.sample(images, k=time_steps))
#
#     else:
#         # test uniform sampling
#         if (len(images) > time_steps):
#             images = sorted(sampling(images, time_steps))
#     # print(images)
#     i = np.random.randint(0, 30)
#     j = np.random.randint(0, 30)
#
#     brightness = 1 + random.uniform(-0.1, +0.1)
#     contrast = 1 + random.uniform(-0.1, +0.1)
#     hue = random.uniform(0, 1) / 20.0
#
#     r_resize = ((256, 256))
#
#     # brightness = 1
#     # contrast = 1
#     # hue = 0
#     gamma = 1. + random.uniform(-0.25, +0.25)
#     grayscale = torchvision.transforms.Grayscale(num_output_channels=3) if (
#             random.uniform(0., 1.) > 0.8) else torchvision.transforms.RandomGrayscale(
#         p=0.0)  # torchvision.transforms.RandomGrayscale(p=0.2)
#
#     t1 = VideoRandomResizedCrop(dim[0], scale=(0.9, 1.0), ratio=(0.8, 1.2))
#
#     px = cv2.cvtColor(cv2.imread(images[0]), cv2.COLOR_BGR2RGB)
#     px = px.reshape((-1, 3)) / 255.
#     px -= px.mean(axis=0)
#     weight, vec = np.linalg.eig(np.cov(px.T))
#
#     for img_path in images:
#         img_tensor = cv2.resize(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB), (224, 224))
#         img_sequence.append(img_tensor)
#
#     pad_len = time_steps - len(images)
#     imgs = np.stack(img_sequence, axis=0)
#     # print(imgs.shape)
#
#     ims_src, imss_tensors = im_augmentation(imgs, weight, vec)
#     # print(ims_src,np.array(ims_src, dtype=np.uint8))
#     # img1 = t(img).permute(1,2,0).numpy()
#     # import cv2
#     # for i in range(ims_src.shape[0]):
#     #
#     #     cv2.imshow('dsd',cv2.cvtColor(ims_src[i],cv2.COLOR_RGB2BGR))
#     #     cv2.waitKey(100)
#     X1 = imss_tensors
#     #
#     if (padding):
#         X1 = pad_video(X1, padding_size=pad_len, padding_type='zeros')
#     elif (len(images) < 16):
#         X1 = pad_video(X1, padding_size=16 - len(images), padding_type='zeros')
#     # if augmentation =='train':
#     #     X1 = video_tensor_shuffle(X1)
#
#     return X1
