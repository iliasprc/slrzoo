import glob
import os
import random

import numpy as np
import torchvision
from torch.utils.data import Dataset

from datasets.loader_utils import im_augmentation
from datasets.loader_utils import multi_label_to_index_out_of_vocabulary
from datasets.loader_utils import pad_video, sampling, VideoRandomResizedCrop
from utils.utils import load_csv_file

dataset_path = '/home/papastrat/Desktop/ilias/datasets/'
ssd_path = ''
phv1_path = '/home/andrster/Desktop/scripts/Datasets/phoenix2014-release/phoenix-2014-multisigner/features/fullFrame-210x260px/'
phv1_path = '/media/tomastheod/Ssd/phoenix_version1/phoenix2014-release/phoenix-2014-multisigner/features/fullFrame-210x260px/'
phv1_path = '/home/papastrat/Desktop/ilias/datasets/phoenix2014-release/phoenix-2014-multisigner/features/fullFrame-210x260px/'


# torch.manual_seed(SEED)
# np.random.seed(SEED)
# random.seed(SEED)
#
# torch.cuda.manual_seed(SEED)


class PH2014_CUI_AUG(Dataset):
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
        filepath = './files/phoenix2014/' + mode + '_phoenixv1.csv'
        print(self.mode, filepath)
        self.list_IDs, self.labels = load_csv_file(filepath)
        self.classes = classes
        self.seq_length = args.seq_length

        self.mode = mode
        if (modality == 'full'):
            self.images_path = phv1_path + mode + '/'
        elif (modality == 'hand'):
            # dim = (112,112)

            self.images_path = hands_path + mode + '/'

        print("Modality used is ", self.images_path)
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

        y = multi_label_to_index_out_of_vocabulary(classes=self.classes, target_labels=self.labels[index])

        x = self.load_video_sequence(path=os.path.join(self.images_path, ID), time_steps=self.seq_length, dim=self.dim,
                                     augmentation=self.mode, padding=self.padding, normalize=self.normalize,
                                     img_type='png')

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
        # if augmentation =='train':
        #     X1 = video_tensor_shuffle(X1)

        return X1
