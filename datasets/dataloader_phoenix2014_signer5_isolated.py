import os

from torch.utils.data import Dataset

from datasets.loader_utils import class2indextensor, load_video_sequence, read_ph2014_isolated

dataset_path = '/home/papastrat/Desktop/ilias/datasets/phoenix2014_isolated_train/'
hand = 'HAND'

train_prefix = "train"
dev_prefix = "test"
dev_filepath = "../files/phoenix_signer5/phoenix_si5_isolated_test.txt"
train_filepath = "../files/phoenix_signer5/phoenix_iso_si5_train.txt"


class PHOENIX_I5_ISO(Dataset):
    def __init__(self, args, mode, classes, dim=(224, 224), modality=None):
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

        ### TO DO READ FILES IN HERE
        list_IDs, labels = read_ph2014_isolated(train_filepath)
        self.classes = classes
        self.seq_length = args.seq_length
        self.labels = labels
        self.list_IDs = list_IDs
        self.mode = mode

        self.images_path = dataset_path

        self.dim = dim
        self.normalize = args.normalize
        self.padding = args.padding

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        ID = os.path.join(dataset_path, self.list_IDs[index])
        # print(ID)
        y = class2indextensor(classes=self.classes, target_label=self.labels[index])
        x = load_video_sequence(path=ID, time_steps=self.seq_length, dim=self.dim,
                                augmentation=self.mode, padding=self.padding, normalize=self.normalize, img_type='png')

        return x, y
