import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import glob
import os
import random
from omegaconf import OmegaConf
import numpy as np
import torch
from PIL import Image
from base.base_loader import BaseDataset
from datasets.loader_utils import multi_label_to_index
from datasets.loader_utils import pad_video, video_transforms, sampling, VideoRandomResizedCrop, read_gsl_continuous, \
    gsl_context, read_bounding_box

feats_path = 'gsl_cont_features/'
train_prefix = "train"
dev_prefix = "dev"
test_prefix = "test"
train_filepath = "files/GSL_continuous/gsl_split_SI_train.csv"
dev_filepath = "files/GSL_continuous/gsl_split_SI_dev.csv"
test_filepath = "files/GSL_continuous/gsl_split_SI_test.csv"


class GSL_SI(BaseDataset):
    def __init__(self, config, args, mode, classes):
        """

        Args:
            config:
            args:
            mode:
            classes:
        """
        super(GSL_SI, self).__init__(config, args, mode, classes)

        cwd_path = args.cwd
        config = OmegaConf.load(os.path.join(args.cwd, "datasets/gsl_si/dataset.yml"))['dataset']
        self.modality = config.modality
        self.mode = mode
        self.dim = config.dim
        self.num_classes = config.classes
        self.seq_length = config[self.mode]['seq_length']
        self.normalize = config.normalize
        self.padding = config.padding
        self.augmentation = config[self.mode]['augmentation']
        self.return_context = self.args.return_context
        if self.mode == train_prefix:
            self.list_IDs, self.list_glosses = read_gsl_continuous(os.path.join(args.cwd, train_filepath))

        elif self.mode == dev_prefix:
            self.list_IDs, self.list_glosses = read_gsl_continuous(os.path.join(args.cwd, dev_filepath))

        elif self.mode == test_prefix:
            self.list_IDs, self.list_glosses = read_gsl_continuous(os.path.join(args.cwd, test_filepath))

        print(f"{len(self.list_IDs)} {self.mode} instances")

        self.bbox = read_bounding_box(os.path.join(args.cwd, 'files/GSL_continuous/bbox_for_gsl_continuous.txt'))

        self.context = gsl_context(self.list_IDs, self.list_glosses)

        if (self.modality == 'full'):
            self.data_path = os.path.join(self.args.input_data, config.images_path)
            self.get = self.video_loader
        elif (self.modality == 'features'):
            self.data_path = os.path.join(self.args.input_data, config.features_path)

            self.get = self.feature_loader

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        return self.get(index)

    def feature_loader(self, index):
        folder_path = os.path.join(self.data_path, self.list_IDs[index])
        # print(folder_path)

        y = multi_label_to_index(classes=self.classes, target_labels=self.list_glosses[index])
        if self.context[index] != None:

            c = multi_label_to_index(classes=self.classes, target_labels=self.context[index])
        else:
            c = torch.tensor([0], dtype=torch.int)
        x = torch.FloatTensor(np.load(folder_path + '.npy')).squeeze(0)
        if self.return_context:
            return x, [y, c]
        return x, y

    def video_loader(self, index):

        x = self.load_video_sequence(path=self.list_IDs[index],
                                     img_type='jpg')
        y = multi_label_to_index(classes=self.classes, target_labels=self.list_glosses[index])

        return x, y

    def load_video_sequence(self, path,
                            img_type='png'):

        images = sorted(glob.glob(os.path.join(self.data_path, path, ) + '/*' + img_type))

        h_flip = False
        img_sequence = []
        # print(images)
        if (len(images) < 1):
            print(os.path.join(self.data_path, path))
        bbox = self.bbox.get(path)

        if (self.augmentation):
            ## training set temporal  AUGMENTATION
            temporal_augmentation = int((np.random.randint(80, 100) / 100.0) * len(images))
            if (temporal_augmentation > 15):
                images = sorted(random.sample(images, k=temporal_augmentation))
            if (len(images) > self.seq_length):
                # random frame sampling
                images = sorted(random.sample(images, k=self.seq_length))

        else:
            # test uniform sampling
            if (len(images) > self.seq_length):
                images = sorted(sampling(images, self.seq_length))

        i = np.random.randint(0, 30)
        j = np.random.randint(0, 30)
        brightness = 1 + random.uniform(-0.2, +0.2)
        contrast = 1 + random.uniform(-0.2, +0.2)
        hue = random.uniform(0, 1) / 10.0
        r_resize = ((256, 256))
        crop_or_bbox = random.uniform(0, 1) > 0.5
        to_flip = random.uniform(0, 1) > 1
        grayscale = random.uniform(0, 1) > 0.9
        t1 = VideoRandomResizedCrop(self.dim[0], scale=(0.9, 1.0), ratio=(0.8, 1.2))
        for img_path in images:

            frame_o = Image.open(img_path)
            frame_o.convert('RGB')

            crop_size = 120
            ## CROP BOUNDING BOX
            ## CROP BOUNDING BOX

            frame1 = np.array(frame_o)

            frame1 = frame1[:, crop_size:648 - crop_size]
            frame = Image.fromarray(frame1)

            if self.augmentation:

                ## training set DATA AUGMENTATION

                frame = frame.resize(r_resize)

                img_tensor = video_transforms(img=frame, i=i, j=j, bright=brightness, cont=contrast, h=hue,
                                              dim=self.dim,
                                              resized_crop=t1,
                                              augmentation=True,
                                              normalize=self.normalize, crop=crop_or_bbox, to_flip=to_flip,
                                              grayscale=grayscale)
                img_sequence.append(img_tensor)
            else:
                # TEST set  NO DATA AUGMENTATION
                frame = frame.resize(self.dim)

                img_tensor = video_transforms(img=frame, i=i, j=j, bright=1, cont=1, h=0, dim=self.dim,
                                              augmentation=False,
                                              normalize=self.normalize)
                img_sequence.append(img_tensor)
        pad_len = self.seq_length - len(images)

        X1 = torch.stack(img_sequence).float()

        if (self.padding):
            X1 = pad_video(X1, padding_size=pad_len, padding_type='zeros')
        if (len(images) < 25):
            X1 = pad_video(X1, padding_size=25 - len(images), padding_type='zeros')
        return X1
