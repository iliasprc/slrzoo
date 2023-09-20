import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self, config, args, mode, classes):
        """

        Args:
            config:
            args:
            mode:
            classes:
        """
        super(BaseDataset, self).__init__()

        self.args = args
        self.classes = classes
        self.cwd_path = args.cwd
        self.mode = mode
        self.list_IDs, self.list_glosses = [], []

        self.config = config

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        """
        Basically, __getitem__ of a torch dataset.
        Args:
            idx (int): Index of the sample to be loaded.
        """

        raise NotImplementedError

    def feature_loader(self, index):
        """


        Args:
            index:
        """
        raise NotImplementedError

    def video_loader(self, index):
        """


        Args:
            index:
        """
        raise NotImplementedError
