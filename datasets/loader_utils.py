import csv
import glob
import math
import os
import pickle
import random

import cv2
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
from torchvision import transforms

from utils.utils import load_config


def select_isolated_dataset(config,args):
    dim = (224, 224)

    test_params = {'batch_size': config.dataset.train.batch_size,
                   'shuffle': False,
                   'num_workers': 2}

    train_params = {'batch_size':config.dataset.validation.batch_size,
                    'shuffle': True,
                    'num_workers': 2}
    config = config.dataset

    if (args.dataset == 'phoenix_iso'):

        print("RUN ON PHOENIX 2014 ISOLATED")
        train_prefix = "train"
        test_prefix = "test"
        classes, indices = read_phoenix_2014_classes(os.path.join(args.cwd, 'files/phoenix2014/classes.txt'))
        print('Number of Classes {} \n \n  '.format(len(classes)))

        from datasets import PHOENIX2014_ISO
        training_set = PHOENIX2014_ISO(dataset_config, args, train_prefix, classes)
        training_generator = data.DataLoader(training_set, **train_params)

        test_set = PHOENIX2014_ISO(dataset_config, args, test_prefix, classes)
        test_generator = data.DataLoader(test_set, **test_params)
        return training_generator, test_generator, None, classes
    elif (args.dataset == 'phoenix_iso_cui'):

        print("RUN ON PHOENIX 2014 CUI ALIGNMENTS ISOLATED")
        train_prefix = "train"
        test_prefix = "test"
        classes, indices = read_phoenix_2014_classes(os.path.join(args.cwd, 'files/phoenix2014/classes.txt'))
        print('Number of Classes {} \n \n  '.format(len(classes)))

        from datasets import PH2014_ISO_CUI
        training_set = PH2014_ISO_CUI(args, train_prefix, classes, dim)
        training_generator = data.DataLoader(training_set, **train_params)

        test_set = PH2014_ISO_CUI(args, test_prefix, classes, dim)
        test_generator = data.DataLoader(test_set, **test_params)
        return training_generator, test_generator, None, classes
    elif (args.dataset == "phoenix_iso_I5"):
        train_prefix = "train"
        test_prefix = "test"
        classes, indices = read_phoenix_2014_classes(args.cwd.joinpath('files/phoenix2014/classes.txt'))
        print('Number of Classes {} \n \n  '.format(len(classes)))

        from datasets import PHOENIX_I5_ISO
        training_set = PHOENIX_I5_ISO(args, train_prefix, classes, dim)
        training_generator = data.DataLoader(training_set, **train_params)

        test_set = PHOENIX_I5_ISO(args, test_prefix, classes, dim)
        test_generator = data.DataLoader(test_set, **test_params)
        return training_generator, test_generator, classes

    elif (args.dataset == "ms_asl"):
        N = 100
        print("Run MS-ASL {} classes".format(N))
        from datasets import MSASL_Dataset
        train_prefix = "train"
        test_prefix = "test"
        train_path = '/home/papastrat/Desktop/ilias/datasets/MS_ASL/MS-ASL_annotations/MSASL_train.json'

        _, classes = select_ASL_subset(train_path, 'TRAIN', N)
        training_set = MSASL_Dataset(mode=train_prefix, dim=dim, classes=N)
        training_generator = data.DataLoader(training_set, **train_params)

        test_set = MSASL_Dataset(mode=test_prefix, dim=dim, classes=N)
        test_generator = data.DataLoader(test_set, **test_params)
        return training_generator, test_generator, classes



    elif (args.dataset == 'gsl_iso'):

        print("RUN ON GREEK ISOLATED")
        train_prefix = "train"
        test_prefix = "test"
        indices, classes, id2w = read_gsl_isolated_classes(args.cwd.joinpath('files/GSL_isolated/iso_classes.csv'))
        print('Number of Classes {} \n \n  '.format(len(classes)))

        from datasets import GSL_ISO
        training_set = GSL_ISO(args, train_prefix, classes, dim)
        training_generator = data.DataLoader(training_set, **train_params)

        val_set = GSL_ISO(args, test_prefix, classes, dim)
        val_generator = data.DataLoader(val_set, **test_params)

        test_set = GSL_ISO(args, 'augment', classes, dim)
        test_generator = data.DataLoader(test_set, **test_params)
        return training_generator, val_generator, test_generator, classes
    elif args.dataset == 'csl_iso':

        train_prefix = "train"
        test_prefix = "test"

        indices, classes, id2w = read_csl_iso_classes(
            args.cwd.joinpath('files/csl_isolated/dictionary_chineze_isolated.txt'))
        from datasets import CSL_ISO

        training_set = CSL_ISO(args, train_prefix, classes, dim)
        training_generator = data.DataLoader(training_set, **train_params)

        test_set = CSL_ISO(args, test_prefix, classes, dim)
        test_generator = data.DataLoader(test_set, **test_params)

        return training_generator, test_generator, classes
    else:
        from datasets.dummy_dataset.islr_dataset import RandomISLRdataset
        classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        training_set = RandomISLRdataset(config,args, 'train',classes)
        training_generator = data.DataLoader(training_set, **train_params)

        val_set = RandomISLRdataset(config,args, 'validation',classes)
        validation_generator = data.DataLoader(val_set, **train_params)
        dict_ = dict(zip(classes,classes))
        return training_generator, validation_generator,None, classes

def select_sc_filse(scen):
    if scen == 'health':
        train_filepath = "files/GSL_continuous/yphresies/health/health_train_SI.txt"
        dev_filepath = "files/GSL_continuous/yphresies/health/health_dev_SI.txt"
        test_filepath = "files/GSL_continuous/yphresies/health/health_test_SI.txt"
        class_path = 'files/GSL_continuous/yphresies/health/health_classes.txt'
    elif scen == 'police':
        train_filepath = "files/GSL_continuous/yphresies/police/police_train_SI.txt"
        dev_filepath = "files/GSL_continuous/yphresies/police/police_dev_SI.txt"
        test_filepath = "files/GSL_continuous/yphresies/police/police_test_SI.txt"
        class_path = 'files/GSL_continuous/yphresies/police/police_classes.txt'
    elif scen == 'kep':
        train_filepath = "files/GSL_continuous/yphresies/kep/kep_train_SI.txt"
        dev_filepath = "files/GSL_continuous/yphresies/kep/kep_dev_SI.txt"
        test_filepath = "files/GSL_continuous/yphresies/kep/kep_test_SI.txt"
        class_path = 'files/GSL_continuous/yphresies/kep/kep_classes.txt'
    return train_filepath, dev_filepath, test_filepath, class_path


def select_scenario_for_training(args):
    dim = (224, 224)
    test_params = {'batch_size': args.batch_size,
                   'shuffle': False,
                   'num_workers': 2}

    train_params = {'batch_size': args.batch_size,
                    'shuffle': True,
                    'num_workers': 2}

    scenario = ['kep', 'police', 'health']

    train_filepath, dev_filepath, test_filepath, class_path = select_sc_filse(scenario[2])
    train_prefix = "train"
    dev_prefix = "dev"
    test_prefix = "test"
    train_filepath = args.cwd.parent.joinpath(train_filepath)
    dev_filepath = args.cwd.parent.joinpath(dev_filepath)
    test_filepath = args.cwd.parent.joinpath(test_filepath)
    indices, classes, id2w = read_gsl_continuous_classes(args.cwd.parent.joinpath(class_path))
    w2id = {v: k for k, v in id2w.items()}
    from datasets.dataloader_greek_scenaria import KENG
    training_set = KENG(args, train_prefix, classes, train_filepath, dev_filepath, test_filepath, dim)
    training_generator = data.DataLoader(training_set, **train_params)
    validation_set = KENG(args, dev_prefix, classes, train_filepath, dev_filepath, test_filepath, dim)
    validation_generator = data.DataLoader(validation_set, **test_params)
    test_set = KENG(args, test_prefix, classes, train_filepath, dev_filepath, test_filepath, dim)
    test_generator = data.DataLoader(test_set, **test_params)
    return training_generator, validation_generator, test_generator, classes, id2w, w2id
    # elif (args.dataset == 'GSL_SD'):
    #     train_prefix = "train"
    #     dev_prefix = "dev"
    #     test_prefix = "test"
    #     indices, classes, id2w = read_gsl_continuous_classes('./files/GSL_continuous/classes.csv')
    #     w2id = {v: k for k, v in id2w.items()}
    #     from data_loader.dataloader_greek_unseen_split import GSL_SD
    #     training_set = GSL_SD(args, train_prefix, classes, dim)
    #     training_generator = data.DataLoader(training_set, **train_params)
    #     validation_set = GSL_SD(args, dev_prefix, classes, dim)
    #     validation_generator = data.DataLoader(validation_set, **test_params)
    #     test_set = GSL_SD(args, test_prefix, classes, dim)
    #     test_generator = data.DataLoader(test_set, **test_params)
    # return training_generator, validation_generator, test_generator, classes, id2w, w2id


def select_continouous_dataset(args):
    dataset_config = load_config(os.path.join(args.cwd, 'datasets/dataloader.yml'))

    dim = (224, 224)
    test_params = {'batch_size': args.batch_size,
                   'shuffle': False,
                   'num_workers': 2}

    train_params = {'batch_size': args.batch_size,
                    'shuffle': False,
                    'num_workers': 2}
    if (args.dataset == 'phoenix2014_2stream'):
        train_prefix = "train"
        dev_prefix = "dev"
        test_prefix = "test"
        p = 'files/phoenix2014/classes.txt'

        classes, indices = read_phoenix_2014_classes(os.path.join(args.cwd, p))
        print('Number of Classes {} from file {} \n \n  '.format(len(classes), p))
        id2w = dict(zip(indices, classes))

        from datasets.phoenix2014.dataloader_2stream_phoenix2014 import Phoenix2014_2stream_Dataset
        training_set = Phoenix2014_2stream_Dataset(train_prefix, classes, dim)
        training_generator = data.DataLoader(training_set, **train_params)

        validation_set = Phoenix2014_2stream_Dataset(dev_prefix, classes, dim)
        validation_generator = data.DataLoader(validation_set, **test_params)

        test_set = Phoenix2014_2stream_Dataset(test_prefix, classes, dim)
        test_generator = data.DataLoader(test_set, **test_params)

        print(len(id2w))
        # for i in id2w:
        #     print(i)
        id2w[len(id2w)] = 'SOS'
        id2w[len(id2w)] = 'EOS'
        classes.append("SOS")
        classes.append("EOS")
        # for k in id2w:
        #     print(k,id2w[k])

        # print(id2w)
        # print(id2w['EOS'])
        w2id = {v: k for k, v in id2w.items()}

        return training_generator, validation_generator, test_generator, classes, id2w, w2id



    elif (args.dataset == 'phoenix2014'):
        train_prefix = "train"
        dev_prefix = "dev"
        test_prefix = "test"
        p = 'files/phoenix2014/classes.txt'
        classes, indices = read_phoenix_2014_classes(os.path.join(args.cwd, p))
        print('Number of Classes {} from file {} \n \n  '.format(len(classes), p))
        id2w = dict(zip(indices, classes))

        from datasets.phoenix2014.dataloader_phoenix2014 import PHOENIX2014
        training_set = PHOENIX2014(config={}, args=args, mode=train_prefix, classes=classes)
        training_generator = data.DataLoader(training_set, **train_params)

        validation_set = PHOENIX2014(config={}, args=args, mode=dev_prefix, classes=classes)
        validation_generator = data.DataLoader(validation_set, **test_params)

        test_set = PHOENIX2014(config={}, args=args, mode=test_prefix, classes=classes)
        test_generator = data.DataLoader(test_set, **test_params)

        print(len(id2w))
        # for i in id2w:
        #     print(i)
        id2w[len(id2w)] = 'SOS'
        id2w[len(id2w)] = 'EOS'
        classes.append("SOS")
        classes.append("EOS")
        # for k in id2w:
        #     print(k,id2w[k])

        print(id2w)
        # print(id2w['EOS'])
        w2id = {v: k for k, v in id2w.items()}

        return training_generator, validation_generator, test_generator, classes, id2w, w2id


    elif (args.dataset == 'phoenix2014T'):
        train_prefix = "train"
        dev_prefix = "dev"
        test_prefix = "test"
        p = args.cwd.joinpath('files/phoenix2014T/classes.txt')
        classes, indices = read_phoenix_2014T_classes(p)
        print('Number of Classes {} from file {} \n \n  '.format(len(classes), p))
        id2w = dict(zip(indices, classes))
        # print(classes)
        from datasets.phoenix2014T.dataloader_phoenix2014_T import PHOENIX2014T
        training_set = PHOENIX2014T(args, train_prefix, classes, dim, modality=args.modality)
        training_generator = data.DataLoader(training_set, **train_params)

        validation_set = PHOENIX2014T(args, dev_prefix, classes, dim, modality=args.modality)
        validation_generator = data.DataLoader(validation_set, **test_params)

        test_set = PHOENIX2014T(args, test_prefix, classes, dim, modality=args.modality)
        test_generator = data.DataLoader(test_set, **test_params)

        # print(len(id2w))
        # for i in id2w:
        #     print(i)
        id2w[len(id2w)] = 'SOS'
        id2w[len(id2w)] = 'EOS'
        classes.append("SOS")
        classes.append("EOS")
        # for k in id2w:
        #     print(k,id2w[k])

        # print(id2w)
        # print(id2w['EOS'])
        w2id = {v: k for k, v in id2w.items()}

        return training_generator, validation_generator, test_generator, classes, id2w, w2id

    elif (args.dataset == 'phoenix2014_cui_aug'):
        train_prefix = "train"
        dev_prefix = "dev"
        test_prefix = "test"
        p = args.cwd.joinpath('files/phoenix2014/classes.txt')
        classes, indices = read_phoenix_2014_classes(p)
        print('Number of Classes {} from file {} \n \n  '.format(len(classes), p))
        id2w = dict(zip(indices, classes))

        from datasets.phoenix2014.dataloader_phoenix2014_cui_augmentations import PH2014_CUI_AUG
        training_set = PH2014_CUI_AUG(train_prefix, classes, dim, modality=args.modality)
        training_generator = data.DataLoader(training_set, **train_params)

        validation_set = PH2014_CUI_AUG(dev_prefix, classes, dim, modality=args.modality)
        validation_generator = data.DataLoader(validation_set, **test_params)

        test_set = PH2014_CUI_AUG(test_prefix, classes, dim, modality=args.modality)
        test_generator = data.DataLoader(test_set, **test_params)

        print(len(id2w))
        # for i in id2w:
        #     print(i)
        id2w[len(id2w)] = 'SOS'
        id2w[len(id2w)] = 'EOS'
        classes.append("SOS")
        classes.append("EOS")
        # for k in id2w:
        #     print(k,id2w[k])

        print(id2w)
        # print(id2w['EOS'])
        w2id = {v: k for k, v in id2w.items()}

        return training_generator, validation_generator, test_generator, classes, id2w, w2id

    elif (args.dataset == 'phoenixI5'):
        train_prefix = "train"
        dev_prefix = "dev"
        test_prefix = "test"
        classes, indices = read_phoenix_2014_classes('./files/phoenix2014/classes.txt')
        print('Number of Classes {} \n \n '.format(len(classes)))
        id2w = dict(zip(indices, classes))

        from datasets.phoenix2014I5.dataloader_phoenix2014_signer5_continuous import PHOENIX_I5
        training_set = PHOENIX_I5(train_prefix, classes, dim, modality=args.modality)
        training_generator = data.DataLoader(training_set, **train_params)

        validation_set = PHOENIX_I5(dev_prefix, classes, dim, modality=args.modality)
        validation_generator = data.DataLoader(validation_set, **test_params)

        test_set = PHOENIX_I5(test_prefix, classes, dim, modality=args.modality)
        test_generator = data.DataLoader(test_set, **test_params)
        return training_generator, validation_generator, test_generator, classes, id2w

    elif (args.dataset == 'csl_split1'):
        train_prefix = "train"

        test_prefix = "test"
        indices, classes, id2w = read_csl_classes(args.cwd.joinpath('files/csl/csl_classes.txt'))

        from datasets.csl_split1.dataloader_csl_split1 import CSL_SPLIT1
        training_set = CSL_SPLIT1(args, train_prefix, classes, dim)
        training_generator = data.DataLoader(training_set, **train_params)
        validation_set = CSL_SPLIT1(args, test_prefix, classes, dim)
        validation_generator = data.DataLoader(validation_set, **test_params)
        w2id = {v: k for k, v in id2w.items()}
        return training_generator, validation_generator, None, classes, id2w, w2id
    elif (args.dataset == 'csl_split2'):
        train_prefix = "train"

        test_prefix = "test"
        indices, classes, id2w = read_csl_classes(args.cwd.joinpath('files/csl/csl_classes.txt'))

        from datasets.csl_split2.dataloader_csl_split2 import CSL_SPLIT2
        training_set = CSL_SPLIT2(args, train_prefix, classes, dim)
        training_generator = data.DataLoader(training_set, **train_params)
        validation_set = CSL_SPLIT2(args, test_prefix, classes, dim)
        validation_generator = data.DataLoader(validation_set, **test_params)
        w2id = {v: k for k, v in id2w.items()}
        return training_generator, validation_generator, None, classes, id2w, w2id
    elif (args.dataset == 'gsl_si'):
        train_prefix = "train"
        dev_prefix = "dev"
        test_prefix = "test"

        indices, classes, id2w = read_gsl_continuous_classes(
            os.path.join(args.cwd, 'files/GSL_continuous/classes.csv'))
        w2id = {v: k for k, v in id2w.items()}
        from datasets.gsl_si.dataloader_gsl_si import GSL_SI
        training_set = GSL_SI(config={}, args=args, mode=train_prefix, classes=classes)
        training_generator = data.DataLoader(training_set, **train_params)
        validation_set = GSL_SI(config={}, args=args, mode=dev_prefix, classes=classes)
        validation_generator = data.DataLoader(validation_set, **test_params)
        test_set = GSL_SI(config={}, args=args, mode=test_prefix, classes=classes)
        test_generator = data.DataLoader(test_set, **test_params)
        return training_generator, validation_generator, test_generator, classes, id2w, w2id
    elif (args.dataset == 'gsl_sd'):
        train_prefix = "train"
        dev_prefix = "dev"
        test_prefix = "test"
        indices, classes, id2w = read_gsl_continuous_classes(
            os.path.join(args.cwd, 'files/GSL_continuous/classes.csv'))
        w2id = {v: k for k, v in id2w.items()}
        from datasets.gsl_sd.dataloader_gsl_sd import GSL_SD
        training_set = GSL_SD(args, train_prefix, classes, dim)
        training_generator = data.DataLoader(training_set, **train_params)
        validation_set = GSL_SD(args, dev_prefix, classes, dim)
        validation_generator = data.DataLoader(validation_set, **test_params)
        test_set = GSL_SD(args, test_prefix, classes, dim)
        test_generator = data.DataLoader(test_set, **test_params)
        return training_generator, validation_generator, test_generator, classes, id2w, w2id
    elif (args.dataset == 'signum_continuous'):

        from datasets.signum.dataloader_signum_continuous import SIGNUM

        train_prefix = "train"
        dev_prefix = "test"
        filepath = args.cwd.joinpath('files/Signum_continuous/database.pkl')
        train_list_IDs, train_labels, train_tokens, id2w, classes, w2id = read_SIGNUM_CONTINUOUS_pkl(filepath,
                                                                                                     mode=train_prefix)

        training_set = SIGNUM(args, train_prefix, classes, w2id, dim)
        training_generator = data.DataLoader(training_set, **train_params)

        validation_set = SIGNUM(args, dev_prefix, classes, w2id, dim)
        validation_generator = data.DataLoader(validation_set, **test_params)

        return training_generator, validation_generator, None, classes, id2w
    else:
        from datasets.dummy_dataset.cslr_dataset import RandomCSLRdataset
        classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        training_set = RandomCSLRdataset(None,args, 'train',classes)
        training_generator = data.DataLoader(training_set, **train_params)

        val_set = RandomCSLRdataset(None,args, 'val',classes)
        validation_generator = data.DataLoader(val_set, **train_params)
        dict_ = dict(zip(classes,classes))
        return training_generator, validation_generator, None, classes, dict_,dict_


def rescale_list(input_list, size):
    assert len(input_list) >= size

    # Get the number to skip between iterations.
    skip = len(input_list) // size

    # Build our new output.
    output = [input_list[i] for i in range(0, len(input_list), skip)]

    # Cut off the last one if needed.
    return output[:size]


def sampling(clip, size):
    return_ind = [int(i) for i in np.linspace(1, len(clip), num=size)]

    return [clip[i - 1] for i in return_ind]


def multi_label_to_index(classes, target_labels):
    indexes = []

    for word in target_labels.strip().split(' '):
        indexes.append(classes.index(word))

    return torch.tensor(indexes, dtype=torch.int)


def multi_label_to_index_out_of_vocabulary(classes, target_labels):
    indexes = []
    for word in target_labels.split(' '):

        if word in classes:
            indexes.append(classes.index(word))

    return torch.tensor(indexes, dtype=torch.int)


def multi_label_to_indexv2(classes, id2w, target_labels):
    indexes = []
    for word in target_labels.split(' '):
        indexes.append(id2w[word])

    return torch.tensor(indexes, dtype=torch.int)


def class2indextensor(classes, target_label):
    indexes = []

    indexes = classes.index(target_label)

    return torch.tensor(indexes, dtype=torch.long)


def pad_video(x, padding_size=0, padding_type='images'):
    if (padding_size != 0):

        if padding_type == 'images':
            pad_img = x[0]

            padx = pad_img.repeat(padding_size, 1, 1, 1)
            X = torch.cat((padx, x))
            return X
        elif padding_type == 'zeros':
            T, C, H, W = x.size()

            padx = torch.zeros((padding_size, C, H, W))
            X = torch.cat((padx, x))
            return X
    return x


def channel_shuffle(x):
    r = x[0, :, :]
    g = x[1, :, :]
    b = x[2, :, :]
    # print(r.shape)
    rgb = [r, g, b]
    random.shuffle(rgb)
    x = torch.stack(rgb, dim=0)

    return x


def video_tensor_shuffle(x):
    # print(x.size())

    r = x[:, 0, :, :]
    g = x[:, 1, :, :]
    b = x[:, 2, :, :]

    # print(test)
    rgb = [r, g, b]
    # print(r[1,0:10,0:10])#,b[1,0:10,0:10],g[1,0:10,0:10])
    random.shuffle(rgb)

    # print(r[1,50:60,50:60],'\n\n',r1[1, 50:60, 50:60])#, b[1, 0:10, 0:10], g[1, 0:10, 0:10])
    x = torch.stack(rgb, dim=1)

    return x


# x = torch.randn(3,4,4)
# print(x)
# x = channel_shuffle(x)
# print(x)

def video_transforms(img, i, j, bright, cont, h, dim=(224, 224), resized_crop=None, grayscale=False, adjust_gamma=None,
                     augmentation=False,
                     normalize=True, crop=False, to_flip=False, verbose=False):
    if (augmentation):
        t = transforms.ToTensor()
        if to_flip:
            img = transforms.functional.hflip(img)

        if crop:
            img = resized_crop(img)
        else:
            img = img.resize(dim)

        # img = transforms.functional.adjust_gamma(img,gamma=adjust_gamma)
        img = transforms.functional.adjust_brightness(img, bright)
        img = transforms.functional.adjust_contrast(img, cont)
        img = transforms.functional.adjust_hue(img, h)
        if grayscale:
            img = transforms.functional.to_grayscale(img, num_output_channels=3)
        if verbose:
            cv2.imshow('IMAGE', cv2.cvtColor(t(img).permute(1, 2, 0).numpy(), cv2.COLOR_RGB2BGR))
            cv2.waitKey(30)
    else:
        t = transforms.ToTensor()
    if (normalize):
        norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    else:

        norm = transforms.Normalize(mean=[0.0, 0.0, 0.0],
                                    std=[1.0, 1.0, 1.0])

    t1 = norm(t(img))
    return t1


def load_video_sequence(path, time_steps, dim=(224, 224), augmentation=False, padding=False, normalize=True,
                        img_type='png'):
    images = sorted(glob.glob(os.path.join(path, '*' + img_type)))

    h_flip = False
    img_sequence = []

    if (augmentation):
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
    t1 = VideoRandomResizedCrop(dim[0], scale=(0.9, 1.0), ratio=(0.8, 1.2))
    for img_path in images:

        frame = Image.open(img_path)
        frame.convert('RGB')
        # print(frame.shape)

        if (augmentation):

            ## training set DATA AUGMENTATION

            frame = frame.resize(r_resize)

            img_tensor = video_transforms(img=frame, i=i, j=j, bright=brightness, cont=contrast, h=hue, dim=dim,
                                          resized_crop=t1,
                                          augmentation=augmentation,
                                          normalize=normalize)
            img_sequence.append(img_tensor)
        else:
            # TEST set  NO DATA AUGMENTATION
            frame = frame.resize(dim)

            img_tensor = video_transforms(img=frame, i=i, j=j, bright=1, cont=1, h=0, dim=dim, augmentation=False,
                                          normalize=normalize)
            img_sequence.append(img_tensor)
    pad_len = time_steps - len(images)

    X1 = torch.stack(img_sequence).float()

    if (padding):
        X1 = pad_video(X1, padding_size=pad_len, padding_type='zeros')
    elif (len(images) < 16):
        X1 = pad_video(X1, padding_size=16 - len(images), padding_type='zeros')
    return X1


def load_video_sequence_uniform_sampling(path, time_steps, dim=(224, 224), augmentation=True, padding=False,
                                         normalize=True,
                                         img_type='png'):
    images = sorted(glob.glob(os.path.join(path, '*' + img_type)))

    img_sequence = []

    if (augmentation):
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

        if (augmentation):

            ## training set DATA AUGMENTATION

            frame = frame.resize(r_resize)

            img_tensor = video_transforms(img=frame, i=i, j=j, bright=brightness, cont=contrast, h=hue, dim=dim,
                                          resized_crop=t1,
                                          augmentation=True,
                                          normalize=normalize)
            img_sequence.append(img_tensor)
        else:
            # TEST set  NO DATA AUGMENTATION
            frame = frame.resize(dim)

            img_tensor = video_transforms(img=frame, i=i, j=j, bright=0, cont=0, h=0, dim=dim, augmentation=False,
                                          normalize=normalize)
            img_sequence.append(img_tensor)
    pad_len = time_steps - len(images)
    if (len(img_sequence) < 1):
        img_sequence.append(torch.zeros((3, dim[0], dim[0])))

        X1 = torch.stack(img_sequence).float()

        X1 = pad_video(X1, padding_size=pad_len - 1, padding_type='zeros')

    elif (padding):
        X1 = torch.stack(img_sequence).float()
        # print(X1.shape)
        X1 = pad_video(X1, padding_size=pad_len, padding_type='zeros')

    return X1


class VideoRandomResizedCrop(object):
    """Crop the given PIL Image to random size and aspect ratio.

    A crop of random size (default: of 0.08 to 1.0) of the original size and a random
    aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.

    Args:
        size: expected output size of each edge
        scale: range of size of the origin size cropped
        ratio: range of aspect ratio of the origin aspect ratio cropped
        interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=Image.BILINEAR):
        if isinstance(size, tuple):
            self.size = size
        else:
            self.size = (size, size)

        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio
        self.parameters = self.get_params(self.scale, self.ratio)

    @staticmethod
    def get_params(scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image): Image to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        area = 256 * 256

        for attempt in range(10):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))
            # print(aspect_ratio,target_area)
            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if w <= 256 and h <= 256:
                i = random.randint(0, 256 - h)
                j = random.randint(0, 256 - w)
                # i = np.random.randint(0, 30)
                # j = np.random.randint(0, 30)
                return i, j, h, w

        # Fallback to central crop
        in_ratio = 256 / 256
        if (in_ratio < min(ratio)):
            w = 256
            h = w / min(ratio)
        elif (in_ratio > max(ratio)):
            h = 256
            w = h * max(ratio)
        else:  # whole image
            w = 256
            h = 256
        i = (256 - h) // 2
        j = (256 - w) // 2

        return i, j, h, w

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped and resized.

        Returns:
            PIL Image: Randomly cropped and resized image.
        """
        i, j, h, w = self.parameters
        return transforms.functional.resized_crop(img, i, j, h, w, self.size, self.interpolation)


def im_augmentation(ims_src, weight, vec, trans=0.1, color_dev=0.2, distortion=True, augmentation='test'):
    num, W, H, _ = ims_src.shape
    if (augmentation == 'train'):
        if distortion:
            ran_noise = np.random.random((4, 2))
            ran_color = np.random.randn(3, )
        else:
            ran_noise = np.ones((4, 2)) * 0.5
            ran_color = np.zeros(3, )

        # # perspective translation
        dst = np.float32([[0., 0.], [1., 0.], [0., 1.], [1., 1.]]) * np.float32([W, H])
        noise = trans * ran_noise * np.float32([[1., 1.], [-1., 1.], [1., -1.], [-1., -1.]]) * [W, H]
        src = np.float32(dst + noise)
        #
        mat = cv2.getPerspectiveTransform(src, dst)
        for i in range(num):
            ims_src[i] = cv2.warpPerspective(ims_src[i], mat, (W, H))

        # color deviation
        deviation = np.dot(vec, (color_dev * ran_color * weight)) * 255.
        # print('DEVIATION!!!!!!!!!!!!!!!!!!!!11 \n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n',deviation)
        # ims_src += deviation[None, None, None, :]
        ims_src = np.add(ims_src, deviation)  # /255.0
        # print('ims ',ims_src.shape)

    ims_tensor = torch.tensor(ims_src / 255.0, dtype=torch.float32)

    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
    ims_tensor = ((ims_tensor - mean) / std).permute(0, 3, 1, 2)
    ims_src = np.array(ims_src, dtype=np.uint8)

    return ims_src, ims_tensor


def read_phoenix_2014T_classes(path):
    classes = []
    indexes = []
    with open(path) as fin:
        reader = csv.reader(fin)
        data = list(reader)
    for idx, item in enumerate(data):
        # print(item[0])
        classes.append(item[0])
        indexes.append(idx)
    return classes, indexes


def read_phoenix_2014_classes(path):
    classes = []
    indexes = []
    with open(path) as fin:
        reader = csv.reader(fin)
        data = list(reader)
    for item in data:
        c, i = item[0].split(' ')
        classes.append(c)
        indexes.append(int(i))
    return classes, indexes


def read_gsl_isolated(csv_path):
    paths, glosses_list = [], []
    classes = []
    data = open(csv_path, 'r').read().splitlines()
    for item in data:
        if (len(item.split('|')) < 2):
            print("\n {} {} {} !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1\n".format(item, path, csv_path))
        path, gloss = item.split('|')

        paths.append(path)

        glosses_list.append(gloss)

    return paths, glosses_list


def read_bounding_box(path):
    bbox = {}
    data = open(path, 'r').read().splitlines()
    for item in data:
        # p#rint(item)
        if (len(item.split('|')) < 2):
            print(item.split('|'))
            print("\n {} {} {} !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1\n".format(item, path, path))
        path, coordinates = item.split('|')
        coords = coordinates.split(',')
        # print(coords)
        x1, x2, y1, y2 = int(coords[0].split(':')[-1]), int(coords[1].split(':')[-1]), int(
            coords[2].split(':')[-1]), int(coords[3].split(':')[-1])
        # print(x1,x2,y1,y2)
        ks = {'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2}

        bbox[path] = ks
        # bbox[path]['x2'] = x2
        # bbox[path]['y1'] = y1
        # bbox[path]['y2'] = y2

        # print(a)

    # bbox.append(a)
    return bbox


def read_bounding_box_continuous(path):
    bbox = {}
    data = open(path, 'r').read().splitlines()
    for item in data:
        # p#rint(item)
        if (len(item.split('|')) < 2):
            print(item.split('|'))
            print("\n {} {} {} !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1\n".format(item, path, path))
        path, coordinates = item.split('|')
        coords = coordinates.split(',')
        # print(coords)
        x1, x2, y1, y2 = int(coords[0].split(':')[-1]), int(coords[1].split(':')[-1]), int(
            coords[2].split(':')[-1]), int(coords[3].split(':')[-1])
        # print(x1,x2,y1,y2)
        ks = {'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2}

        bbox[path] = ks
        # bbox[path]['x2'] = x2
        # bbox[path]['y1'] = y1
        # bbox[path]['y2'] = y2

        # print(a)

    # bbox.append(a)
    return bbox


def read_gsl_continuous(csv_path):
    paths, glosses_list = [], []
    classes = []
    data = open(csv_path, 'r').read().splitlines()
    for item in data:
        if (len(item.split('|')) < 2):
            print("\n {} {} {} !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1\n".format(item, path, csv_path))
        path, glosses = item.split('|')
        # path = path.replace(' GSL_continuous','GSL_continuous')

        paths.append(path)
        # print(path)

        glosses_list.append(glosses)
    return paths, glosses_list


def read_gsl_prev_sentence(paths, ):
    allpaths, prev_sentences = read_gsl_continuous(
        '/home/iliask/PycharmProjects/SLR_GAN/files/GSL_continuous/context_GSL.txt')
    context_dict = dict(zip(allpaths, prev_sentences))
    context = []
    for idx, path in enumerate(paths):
        context.append(context_dict[path])
    return context


def gsl_context(paths, glosses_list):
    context = []
    prev_ID = -2
    for idx, path in enumerate(paths):
        # print(path)
        sentence_ID = int(path.split('/')[-1].split('sentences')[-1])
        if sentence_ID == prev_ID + 1:
            # print('prev {} now {}'.format(prev_ID,sentence_ID))
            context.append(glosses_list[idx - 1])
        else:
            a = 0
            # print('NOTTT prev {} now {}'.format(prev_ID,sentence_ID))
            context.append(None)
        prev_ID = sentence_ID
        # print(int(sentence_ID))
    assert len(context) == len(glosses_list)
    # print(context[1],' j ',glosses_list[1],' j ',glosses_list[0])
    return context


def read_gsl_isolated_classes(path):
    indices, classes = [], []

    data = open(path, 'r').read().splitlines()
    count = 1
    for d in data:
        label = d

        indices.append(count)
        classes.append(label)
        count += 1

    id2w = dict(zip(indices, classes))

    return indices, classes, id2w


def read_gsl_continuous_classes(path):
    indices, classes = [], []
    classes.append('blank')
    indices.append(0)
    data = open(path, 'r').read().splitlines()
    count = 1
    for d in data:
        label = d

        indices.append(count)
        classes.append(label)
        count += 1

    id2w = dict(zip(indices, classes))

    return indices, classes, id2w


def get_paths_labels(path, split):
    gloss_folders = glob.glob(path + split + '/*')

    labels = []
    for fold in gloss_folders:
        label = fold.split('_')[-1]
        labels.append(label)

    classes = list(set(labels))
    print('Found {} classes in dataset {} split'.format(len(classes), split))
    print("{} examples {} labels".format(len(gloss_folders), len(labels)))
    return gloss_folders, labels


import json


def load_json(path, save_location):
    id2w = dict()
    classes = []
    mode = save_location.lower()

    video_paths = glob.glob(save_location + '/*')

    with open(path, 'r') as jf:
        data = json.load(jf)
        folders = []
        for idx, line in enumerate(data):
            video_url = line['url']
            signer = line['signer_id']
            height = line['height']
            width = line['width']
            label = line['label']
            text_label = line['text']
            fps_json = line['fps']
            start_time = line['start_time']
            end_time = line['end_time']
            start = line['start']
            end = line['end']
            box = line['box']
            vid_name = str(line['url']).split('/')[-1]

            if (text_label not in classes):
                classes.append(text_label)
                id2w[text_label] = label

        print(len(id2w), len(classes))
        return id2w, classes


def select_ASL_subset(path, save_location, N):
    id2w = dict()
    classes = []
    mode = save_location.lower()

    video_paths = glob.glob(save_location + '/*')

    with open(path, 'r') as jf:
        data = json.load(jf)
        folders = []
        for idx, line in enumerate(data):
            video_url = line['url']
            signer = line['signer_id']
            height = line['height']
            width = line['width']
            label = line['label']
            text_label = line['text']
            fps_json = line['fps']
            start_time = line['start_time']
            end_time = line['end_time']
            start = line['start']
            end = line['end']
            box = line['box']
            vid_name = str(line['url']).split('/')[-1]

            if (label < N):
                if (text_label not in classes):
                    classes.append(text_label)
                    id2w[text_label] = label

        print(" Training Subset ASL {}".format(N))
        return id2w, classes


def get_subset_paths_labels(path, split, classes):
    gloss_folders = glob.glob(path + split + '/*')

    examples = []
    labels = []
    for fold in gloss_folders:
        label = fold.split('_')[-1]
        if label in classes:
            labels.append(label)
            examples.append(fold)

    c = list(set(labels))
    print('Found {} classes in dataset {} split'.format(len(c), split))
    print("{} examples {} labels".format(len(examples), len(labels)))
    return examples, labels


def read_csl_iso_classes(path):
    indices, classes = [], []
    data = open(path, 'r').read().splitlines()
    for d in data:
        index, label = d.split('\t')
        indices.append(int(index))
        classes.append(label)

    id2w = dict(zip(indices, classes))

    return indices, classes, id2w


def read_csl_classes(path):
    indices, classes = [], []
    classes.append('blank')
    indices.append(0)
    data = open(path, 'r').read().splitlines()
    for d in data:
        label, index = d.split(',')

        indices.append(int(index) + 1)
        classes.append(label)

    id2w = dict(zip(indices, classes))

    return indices, classes, id2w


def load_phoenix_2014_T(path):
    data_paths = []
    labels = []
    with open(path) as fin:
        reader = csv.reader(fin)
        data = list(reader)
    for item in data:
        path = item[0].split('|')[0]
        label = item[0].split('|')[-2]
        data_paths.append(path)
        labels.append(label)

    return data_paths, labels


def load_phoenix_signer_independent(path):
    data_paths = []
    labels = []
    with open(path) as fin:
        reader = csv.reader(fin)
        data = list(reader)
    for item in data:
        path = item[0].split('|')[0] + '/1/'
        label = item[0].split('|')[-1]
        data_paths.append(path)
        labels.append(label)

    return data_paths, labels


def read_ph2014_isolated(path):
    paths, labels = [], []
    data = open(path, 'r').read().splitlines()
    for item in data:
        path, label = item.split(' ')
        paths.append(path)
        labels.append(label)
    return paths, labels


def read_ph2014_cui_isolated(path):
    paths, labels = [], []
    starts, ends = [], []
    data = open(path, 'r').read().splitlines()
    for item in data:
        path, label, start_frame, end_frame = item.split('|')
        paths.append(path)
        labels.append(label)
        starts.append(start_frame)
        ends.append(end_frame)
    return paths, labels, starts, ends


def read_SIGNUM_CONTINUOUS_pkl(filepath, mode='train'):
    infile = open(filepath, 'rb')
    new_dict = pickle.load(infile, encoding="utf-8")
    infile.close()
    split = new_dict[mode]
    print("KEYS {} ".format(split.keys()))
    vocab = split['vocabulary']

    tokens = split['token']
    annotation = split['annotation']
    folders = split['folder']
    print(len(tokens), len(annotation), len(folders))

    paths = []
    for fold in folders:
        si, rep, fold = fold.split('-')

        new_name = "s01-p{}/{}".format(rep[-2:], fold)
        paths.append(new_name)

    classes = []
    classes.append('blank')
    for i in vocab:
        classes.append(i)
    indices = list(range(len(classes)))
    id2w = dict(zip(indices, classes))
    w2id = dict(zip(classes, indices))
    return paths, annotation, tokens, id2w, classes, w2id


def read_signum_paths(path):
    data_paths = []
    labels = []
    data = open(path, 'r').read().splitlines()
    for d in data:
        video, label = d.split('\t')
        data_paths.append(video)
        labels.append(label)

    return data_paths, labels


def read_signum_classes(path):
    indices, classes = [], []
    data = open(path, 'r').read().splitlines()
    for d in data:

        index, label = d.split('\t')

        print(index, label)

        if len(label.split('|')) > 1:
            words = label.split('|')
            for word in words:
                indices.append(int(index[1:]))
                classes.append(word)
                print(index, word)
        else:

            indices.append(int(index[1:]))
            classes.append(label)

    id2w = dict(zip(classes, indices))
    indices = list(set(indices))
    return indices, classes, id2w


def read_csl_paths(path):
    data_paths = []
    labels = []
    data = open(path, 'r').read().splitlines()
    for d in data:
        video, label = d.split('\t')
        data_paths.append(video)
        labels.append(label)

    return data_paths, labels
