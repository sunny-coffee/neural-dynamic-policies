import torch
from torch.utils.data import Dataset
from mnist.utils.smnist_loader import MatLoader
import scipy.io as scio
import numpy as np


class TrainDataset(Dataset):
    def __init__(self, data_path, train_inds):
        data_path = './mnist/data/40x40-smnist.mat'
        _, _, _, or_tr = MatLoader.load_data(data_path, load_original_trajectories=True)
        Y = torch.Tensor(np.array(or_tr)[:, :, :2]).float()[:12000]
        self.Y_train = Y[train_inds]
        # print(self.Y_train.shape)

    def __getitem__(self, index):
        return self.Y_train[index,:-1,:], self.Y_train[index,1:,:]

    def __len__(self):
        return self.Y_train.shape[0]


class TestDataset(Dataset):
    def __init__(self, data_path, test_inds):
        data_path = './mnist/data/40x40-smnist.mat'
        _, _, _, or_tr = MatLoader.load_data(data_path, load_original_trajectories=True)
        Y = torch.Tensor(np.array(or_tr)[:, :, :2]).float()[:12000]
        self.Y_test = Y[test_inds]

    def __getitem__(self, index):
        return self.Y_test[index,:-1,:], self.Y_test[index,1:,:]

    def __len__(self):
        return self.Y_test.shape[0]