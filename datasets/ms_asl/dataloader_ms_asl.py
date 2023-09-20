import os

from torch.utils.data import Dataset

from datasets.loader_utils import class2indextensor, load_video_sequence_uniform_sampling, select_ASL_subset, get_subset_paths_labels

dataset_path = '/home/papastrat/Desktop/ilias/datasets/MS_ASL/ms_asl_dataset/'
mode = ['train', 'val', 'test']
train_path = '/home/papastrat/Desktop/ilias/datasets/MS_ASL/MS-ASL_annotations/MSASL_train.json'
val_path = '/home/papastrat/Desktop/ilias/datasets/MS_ASL/MS-ASL_annotations/MSASL_val.json'
test_path = '/home/papastrat/Desktop/ilias/datasets/MS_ASL/MS-ASL_annotations/MSASL_test.json'




#
# torch.manual_seed(SEED)
# np.random.seed(SEED)
# random.seed(SEED)
#
# torch.cuda.manual_seed(SEED)
class MSASL_Dataset(Dataset):
    def __init__(self, args, mode, classes=1000, dim=(224, 224), modality=None):
        """
        Args:
            path_prefix : train or test and path prefix to read frames acordingly
            classes : list of classes
            channels: Number of channels of frames
            seq_length : Number of frames to be loaded in a sample
            dim: Dimensions of the frames
            subset : select 100 or 200 or 500 or 1000 classes
            normalize : normalize tensor with imagenet mean and std
            padding : padding of video to size seq_length

        """

        self.mode = mode

        self.images_path = dataset_path
        self.seq_length = args.seq_length
        self.dim = dim
        self.normalize = True
        self.padding = True

        N = classes
        ## select subset of N classes
        _, self.classes = select_ASL_subset(train_path, 'TRAIN', N)
        self.list_IDs, self.labels = get_subset_paths_labels(dataset_path, self.mode, self.classes)

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        ID = os.path.join(dataset_path, self.list_IDs[index])
        # print(ID)
        y = class2indextensor(classes=self.classes, target_label=self.labels[index])
        x = load_video_sequence_uniform_sampling(path=self.list_IDs[index], time_steps=self.seq_length, dim=self.dim,
                                                 augmentation=self.mode, padding=self.padding, normalize=self.normalize,
                                                 img_type='jpg')

        return x, y
