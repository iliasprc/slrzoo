import csv
import json
import logging
import os
import subprocess
from collections import OrderedDict
from itertools import repeat
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from utils.metrics import wer_generic

logging.captureWarnings(True)


def get_logger(name):
    """Logs a message
    Args:
    name(str): name of logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level=logging.DEBUG)
    return logger


def make_dirs_if_not_present(path):
    """
    creates new directory if not present
    """
    if not os.path.exists(path):
        os.makedirs(path)


def load_config(config_file):
    """

    Args:
        config_file ():

    Returns:

    """
    with open(config_file, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)


def ensure_dir(dirname):
    """
    Args:
        dirname ():
    """
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_json(fname):
    """

    Args:
        fname ():

    Returns:

    """
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    """

    Args:
        content ():
        fname ():
    """
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def inf_loop(data_loader):
    ''' wrapper function for endless data loader.

    Args:
        data_loader ():
    '''
    for loader in repeat(data_loader):
        yield from loader


def check_dir(path):
    if not os.path.exists(path):
        print("Checkpoint Directory does not exist! Making directory {}".format(path))
        os.makedirs(path)


def get_lr(optimizer):
    """

    Args:
        optimizer ():

    Returns:

    """
    for param_group in optimizer.param_groups:
        return param_group['lr']


class MetricTracker:
    def __init__(self, *keys, writer=None, mode='/'):
        """

        Args:
            *keys ():
            writer ():
            mode ():
        """
        self.writer = writer
        self.mode = mode + '/'
        self.keys = keys
        # print(self.keys)
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1, writer_step=1):
        if self.writer is not None:
            self.writer.add_scalar(self.mode + key, value, writer_step)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def update_all_metrics(self, values_dict, n=1, writer_step=1):
        for key in values_dict:
            self.update(key, values_dict[key], n, writer_step)

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)

    def calc_all_metrics(self):
        """
        Calculates string with all the metrics
        Returns:

        """
        s = ''
        d = dict(self._data.average)
        for key in dict(self._data.average):
            s += f'{key} {d[key]:7.4f}\t'

        return s

    def wer(self):
        wer_keys = ['S', 'D', 'I', 'C']
        if wer_keys in self.keys:
            return (self._data.total['S'] + self._data.total['D'] + self._data.total['I']) / (
                    self._data.total['S'] + self._data.total['D'] + self._data.total['C'])
        else:
            return self._data.average['wer']


def evaluate_phoenix2014T(path, mode, save_location='/home/papastrat/Desktop/ilias/phoenixT_eval/'):
    """

    Args:
        path ():
        mode ():
        save_location ():

    Returns:

    """
    avg_wer = 0.0
    if (mode == 'dev'):
        referencefile = '../files/phoenix2014T/PHOENIX-2014-T.dev.corpus.csv'
    else:
        referencefile = '../files/phoenix2014T/PHOENIX-2014-T.test.corpus.csv'
    with open(path, 'r') as predfile:
        preds = predfile.read().splitlines()
    with open(referencefile, 'r') as ref1:
        ref = ref1.read().splitlines()

    folder = path.rsplit('/', 1)[0]

    name = path.split('/')[-1].split('wer')[-1].split('.csv')[0]
    print(len(preds), len(ref))
    paths = []
    eval_name = mode + "_.ctm"
    print(eval_name)
    current_dir = os.getcwd()

    os.chdir(save_location)
    f = open(eval_name, 'w')
    for i in range(len(ref)):
        paths.append(ref[i].split('|')[0])

        words = preds[i].split(' ')[0:-1]
        time = 0
        # print(words,ref[i].split('|')[-2])
        # print(' '.join(words).replace('cl-','').replace('loc-',''))
        if ('RAUM' in ref[i].split('|')[-2]):
            print(ref[i].split('|')[-2])
        for j in words:
            step = np.random.rand() / 2.0
            folder = (ref[i].split('|')[0])
            # print(folder)
            f.write("{} {} {:.3f} {:.3f} {}\n".format(folder, 1, time, step, j))
            time += step

        temp_wer, C, S, I, D = wer_generic(ref[i].split('|')[-2], ' '.join(words))
        avg_wer += temp_wer
        # print(temp_wer)
    f.close()
    # s#ubprocess.run(["bash", "evaluatePhoenix2014.sh", eval_name, mode])
    print(100 * avg_wer / len(ref))
    os.chdir(current_dir)
    return eval_name


def one_hot(target, num_classes):
    """

    Args:
        target ():
        num_classes ():

    Returns:

    """
    labels = target.reshape(-1, 1).cpu()
    one_hot_target = ((labels == torch.arange(num_classes).reshape(1, num_classes)).float())
    return one_hot_target


def smooth_one_hot(true_labels, classes, smoothing=0.1):
    """
    if smoothing == 0, it's one-hot method
    if 0 < smoothing < 1, it's smooth method

    Args:
        true_labels ():
        classes ():
        smoothing ():

    """
    assert 0 <= smoothing < 1
    confidence = 1.0 - smoothing
    label_shape = torch.Size((true_labels.size(0), classes))
    with torch.no_grad():
        true_dist = torch.empty(size=label_shape, device=true_labels.device)
        true_dist.fill_(smoothing / (classes - 1))
        true_dist.scatter_(1, true_labels.data.unsqueeze(1), confidence)
    return true_dist


# target = torch.LongTensor([[2,3,1,4]])
# print(target.shape)
# classes = 5
# print(one_hot(target,classes))
# print(smooth_one_hot(target,classes))
# evaluate_phoenix2014T('/home/papastrat/Desktop/ilias/Sign_Language_Recognition/checkpoints/model_cui
# /dataset_phoenix2014T/date_06_02_2020_12.33.38/test_predictions_epoch6_wer_26.44268912170231_.csv','test')


def evaluate_phoenix(path, mode, evaluation_script_folder='/phoenix_hypothesis/'):
    """

    Args:
        path ():
        mode ():
        evaluation_script_folder ():

    Returns:

    """
    dir = os.getcwd()
    if (mode == 'dev'):
        referencefile = os.path.join('/home/papastrat/PycharmProjects/SLR_GAN-dev/', 'files/phoenix2014/dev_phoenixv1.csv')
    else:
        referencefile = os.path.join('/home/papastrat/PycharmProjects/SLR_GAN-dev/', 'files/phoenix2014/test_phoenixv1.csv')
    with open(path, 'r') as predfile:
        preds = predfile.read().splitlines()

    with open(referencefile, 'r') as ref1:
        ref = ref1.read().splitlines()

    cpkt_folder = path.rsplit('/', 1)[0]

    paths = []
    eval_name = path.split('/')[-1].split('.csv')[0] + "_.ctm"

    os.chdir(evaluation_script_folder)
    f = open(eval_name, 'w')
    for i in range(len(ref)):
        paths.append(ref[i].split(',')[0])

        words = preds[i].split(' ')[0:-1]
        time = 0
        for j in words:
            step = np.random.rand() / 2.0
            folder = (ref[i].split(',')[0]).split('/')[0]

            f.write("{} {} {:.3f} {:.3f} {}\n".format(folder, 1, time, step, j))
            time += step

    f.close()
    result = subprocess.run(["bash", "evaluatePhoenix2014.sh", eval_name, mode], stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)

    print("SDSDSD \n", result.stdout)
    phoenix_wer = float(str(result.stdout).split('%')[0][-6:])
    # print(phoenix_wer)
    listdir = os.listdir(evaluation_script_folder)
    matching = [s for s in listdir if eval_name in s]

    if not os.path.exists(os.path.join(cpkt_folder, 'phoenix_evaluations')):
        os.makedirs(os.path.join(cpkt_folder, 'phoenix_evaluations'))
    for item in matching:
        os.rename(item, os.path.join(cpkt_folder, 'phoenix_evaluations', item))
    os.chdir(dir)
    return phoenix_wer


def load_csv_file(path):
    """

    Args:
        path ():

    Returns:

    """
    data_paths = []
    labels = []
    with open(path) as fin:
        reader = csv.reader(fin)
        data = list(reader)
    for item in data:
        data_paths.append(item[0])
        labels.append(item[1])
    return data_paths, labels


def txt_logger(txtname, log):
    """

    Args:
        txtname ():
        log ():
    """
    with open(txtname, 'a') as f:
        for item in log:
            f.write(item)
            f.write(',')

        f.write('\n')


def write_csv(data, name):
    """

    Args:
        data ():
        name ():
    """
    with open(name, 'w') as fout:
        for item in data:
            # print(item)
            fout.write(item)
            fout.write('\n')
