import os

from torch.utils.data import Dataset

from datasets.loader_utils import load_video_sequence, multi_label_to_index_out_of_vocabulary, load_phoenix_signer_independent

dataset_path = '/home/hatzis/Desktop/teo/'
ssd_path = ''
phv1_path = '/home/papastrat/Desktop/ilias/datasets/phoenix2014-release/phoenix-2014-signerindependent-SI5/features/fullFrame-210x260px/'


class PHOENIX_I5(Dataset):
    def __init__(self, args, mode, classes, dim=(224, 224), modality='full'):
        """
        Args:

            channels: Number of channels of frames
            timeDepth: Number of frames to be loaded in a sample
            xSize, ySize: Dimensions of the frames
            mean: Mean valuse of the training set videos over each channel
        """

        self.mode = mode
        filepath = './files/phoenix_signer5/' + mode + '.SI5.corpus.csv'
        print(self.mode, filepath)
        self.list_IDs, self.labels = load_phoenix_signer_independent(filepath)
        self.classes = classes
        self.seq_length = args.seq_length

        if (modality == 'full'):
            self.images_path = phv1_path + mode + '/'
        elif (modality == 'hand'):
            self.images_path = mode + '/'
        if (mode != 'train'):
            self.mode = 'test'
            print("augmentation {}".format(self.mode))
        print("Modality used is ", self.images_path)

        self.dim = dim

        self.normalize = args.normalize
        self.padding = args.padding

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):

        ID = self.list_IDs[index]
        # print(os.path.join(self.images_path, ID))
        y = multi_label_to_index_out_of_vocabulary(classes=self.classes, target_labels=self.labels[index])

        x = load_video_sequence(path=os.path.join(self.images_path, ID), time_steps=self.seq_length, dim=self.dim,
                                augmentation=self.mode, padding=self.padding, normalize=self.normalize, img_type='png')
        return x, y
