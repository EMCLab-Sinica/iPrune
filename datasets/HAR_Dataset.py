import pathlib
import os
import sys
import functools
import numpy as np
cwd = os.getcwd()
sys.path.append(cwd+'/../')

from torch.utils.data.dataset import Dataset

class HAR_Dataset(Dataset):
    def __init__(self, split):
        if not os.path.isdir('~/.cache/UCI HAR Dataset/'):
            sys.path.append(cwd + '/../../')
            from utils import download_file, extract_archive
            print('Download HAR dataset ...')
            archive_dir = download_file('https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip', filename='UCI HAR Dataset.zip', post_processor=functools.partial(extract_archive, subdir='UCI HAR Dataset'))
        root =  pathlib.Path('~/.cache/UCI HAR Dataset/').expanduser()
        sys.path.append(cwd + '/../../data/deep-learning-HAR/utils')
        from utilities import read_data
        self.imgs, self.labels, self.list_ch_train = read_data(data_path=root, split=split) # train
        # make sure they contain only valid labels [0 ~ class -1]
        self.labels = self.labels - 1

        assert len(self.imgs) == len(self.labels), "Mistmatch in length!"
        # Normalize?
        self.imgs = self.standardize(self.imgs)

    def standardize(self, data):
        """ Standardize data """
        # Standardize train and test
        standardized_data = (data - np.mean(data, axis=0)[None,:,:]) / np.std(data, axis=0)[None,:,:]
        # (batch, 9, 128) => (batch, 9, 1, 128)
        standardized_data = np.expand_dims(standardized_data, axis=2)
        return standardized_data

    def __getitem__(self, index):
        return self.imgs[index], self.labels[index]

    def __len__(self):
        return len(self.imgs)
