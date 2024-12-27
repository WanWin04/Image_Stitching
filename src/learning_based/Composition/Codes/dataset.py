from torch.utils.data import Dataset
import numpy as np
import cv2, torch
import os
import glob
from collections import OrderedDict
import random


class TrainDataset(Dataset):
    def __init__(self, data_path):
        # Initialize dataset with the path to the training data
        self.train_path = data_path
        self.datas = OrderedDict()

        # Load all subdirectories and filter only the required folders
        datas = glob.glob(os.path.join(self.train_path, '*'))
        for data in sorted(datas):
            data_name = data.split(os.sep)[-1]
            if data_name == 'warp1' or data_name == 'warp2' or data_name == 'mask1' or data_name == 'mask2':
                # Store the path and sorted image list for each data type
                self.datas[data_name] = {}
                self.datas[data_name]['path'] = data
                self.datas[data_name]['image'] = glob.glob(os.path.join(data, '*.jpg'))
                self.datas[data_name]['image'].sort()
        print(self.datas.keys())

    def __getitem__(self, index):
        # Load and preprocess warp1 image
        warp1 = cv2.imread(self.datas['warp1']['image'][index])
        warp1 = warp1.astype(dtype=np.float32)
        warp1 = (warp1 / 127.5) - 1.0  # Normalize to [-1, 1]
        warp1 = np.transpose(warp1, [2, 0, 1])  # Rearrange dimensions to [C, H, W]

        # Load and preprocess warp2 image
        warp2 = cv2.imread(self.datas['warp2']['image'][index])
        warp2 = warp2.astype(dtype=np.float32)
        warp2 = (warp2 / 127.5) - 1.0  # Normalize to [-1, 1]
        warp2 = np.transpose(warp2, [2, 0, 1])  # Rearrange dimensions to [C, H, W]

        # Load and preprocess mask1
        mask1 = cv2.imread(self.datas['mask1']['image'][index])
        mask1 = mask1.astype(dtype=np.float32)
        mask1 = np.expand_dims(mask1[:, :, 0], 2) / 255  # Use only one channel and normalize to [0, 1]
        mask1 = np.transpose(mask1, [2, 0, 1])  # Rearrange dimensions to [C, H, W]

        # Load and preprocess mask2
        mask2 = cv2.imread(self.datas['mask2']['image'][index])
        mask2 = mask2.astype(dtype=np.float32)
        mask2 = np.expand_dims(mask2[:, :, 0], 2) / 255  # Use only one channel and normalize to [0, 1]
        mask2 = np.transpose(mask2, [2, 0, 1])  # Rearrange dimensions to [C, H, W]

        # Convert all data to PyTorch tensors
        warp1_tensor = torch.tensor(warp1)
        warp2_tensor = torch.tensor(warp2)
        mask1_tensor = torch.tensor(mask1)
        mask2_tensor = torch.tensor(mask2)

        # Randomly decide whether to exchange warp1 with warp2
        if_exchange = random.randint(0, 1)
        if if_exchange == 0:
            # Return tensors in their original order
            return (warp1_tensor, warp2_tensor, mask1_tensor, mask2_tensor)
        else:
            # Return tensors with warp1 and warp2 exchanged
            return (warp2_tensor, warp1_tensor, mask2_tensor, mask1_tensor)

    def __len__(self):
        # Return the total number of samples
        return len(self.datas['warp1']['image'])


class TestDataset(Dataset):
    def __init__(self, data_path):
        # Initialize dataset with the path to the test data
        self.test_path = data_path
        self.datas = OrderedDict()

        # Load all subdirectories and filter only the required folders
        datas = glob.glob(os.path.join(self.test_path, '*'))
        for data in sorted(datas):
            data_name = data.split(os.sep)[-1]
            if data_name == 'warp1' or data_name == 'warp2' or data_name == 'mask1' or data_name == 'mask2':
                # Store the path and sorted image list for each data type
                self.datas[data_name] = {}
                self.datas[data_name]['path'] = data
                self.datas[data_name]['image'] = glob.glob(os.path.join(data, '*.jpg'))
                self.datas[data_name]['image'].sort()

        print(self.datas.keys())

    def __getitem__(self, index):
        # The loading and preprocessing process is the same as TrainDataset
        warp1 = cv2.imread(self.datas['warp1']['image'][index])
        warp1 = warp1.astype(dtype=np.float32)
        warp1 = (warp1 / 127.5) - 1.0
        warp1 = np.transpose(warp1, [2, 0, 1])

        warp2 = cv2.imread(self.datas['warp2']['image'][index])
        warp2 = warp2.astype(dtype=np.float32)
        warp2 = (warp2 / 127.5) - 1.0
        warp2 = np.transpose(warp2, [2, 0, 1])

        mask1 = cv2.imread(self.datas['mask1']['image'][index])
        mask1 = mask1.astype(dtype=np.float32)
        mask1 = np.expand_dims(mask1[:, :, 0], 2) / 255
        mask1 = np.transpose(mask1, [2, 0, 1])

        mask2 = cv2.imread(self.datas['mask2']['image'][index])
        mask2 = mask2.astype(dtype=np.float32)
        mask2 = np.expand_dims(mask2[:, :, 0], 2) / 255
        mask2 = np.transpose(mask2, [2, 0, 1])

        warp1_tensor = torch.tensor(warp1)
        warp2_tensor = torch.tensor(warp2)
        mask1_tensor = torch.tensor(mask1)
        mask2_tensor = torch.tensor(mask2)

        return (warp1_tensor, warp2_tensor, mask1_tensor, mask2_tensor)

    def __len__(self):
        # Return the total number of samples
        return len(self.datas['warp1']['image'])
