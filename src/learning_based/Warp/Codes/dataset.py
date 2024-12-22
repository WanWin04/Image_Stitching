from torch.utils.data import Dataset
import  numpy as np
import cv2, torch
import os
import glob
from collections import OrderedDict
import random


class TrainDataset(Dataset):
    def __init__(self, data_path):
        # Set the desired image width and height
        self.width = 512
        self.height = 512
        # Initialize dataset path
        self.train_path = data_path
        self.datas = OrderedDict()
        
        # Get all file paths in the training directory
        datas = glob.glob(os.path.join(self.train_path, '*'))
        for data in sorted(datas):
            data_name = data.split(os.sep)[-1]
            # If directory is input1 or input2, load its images
            if data_name == 'input1' or data_name == 'input2':
                self.datas[data_name] = {}
                self.datas[data_name]['path'] = data
                self.datas[data_name]['image'] = glob.glob(os.path.join(data, '*.jpg'))
                self.datas[data_name]['image'].sort()  # Sort the image paths
        print(self.datas.keys())  # Print the dataset keys for debugging


    def __getitem__(self, index):
        # Load image from 'input1'
        input1 = cv2.imread(self.datas['input1']['image'][index])
        input1 = cv2.resize(input1, (self.width, self.height))  # Resize to fixed dimensions
        input1 = input1.astype(dtype=np.float32)
        input1 = (input1 / 127.5) - 1.0  # Normalize to [-1, 1]
        input1 = np.transpose(input1, [2, 0, 1])  # Change channel order to (C, H, W)
        
        # Load image from 'input2'
        input2 = cv2.imread(self.datas['input2']['image'][index])
        input2 = cv2.resize(input2, (self.width, self.height))  # Resize to fixed dimensions
        input2 = input2.astype(dtype=np.float32)
        input2 = (input2 / 127.5) - 1.0  # Normalize to [-1, 1]
        input2 = np.transpose(input2, [2, 0, 1])  # Change channel order to (C, H, W)
        
        # Convert to tensor format for PyTorch
        input1_tensor = torch.tensor(input1)
        input2_tensor = torch.tensor(input2)
        
        # Randomly decide whether to return input1 first or input2 first
        if_exchange = random.randint(0, 1)
        if if_exchange == 0:
            return (input1_tensor, input2_tensor)  # Return as (input1, input2)
        else:
            return (input2_tensor, input1_tensor)  # Return as (input2, input1)


    def __len__(self):
        # Return the total number of images in 'input1'
        return len(self.datas['input1']['image'])


class TestDataset(Dataset):
    def __init__(self, data_path):
        # Set the desired image width and height
        self.width = 512
        self.height = 512
        # Initialize dataset path
        self.test_path = data_path
        self.datas = OrderedDict()
        
        # Get all file paths in the testing directory
        datas = glob.glob(os.path.join(self.test_path, '*'))
        for data in sorted(datas):
            data_name = data.split(os.sep)[-1]
            # If directory is input1 or input2, load its images
            if data_name == 'input1' or data_name == 'input2':
                self.datas[data_name] = {}
                self.datas[data_name]['path'] = data
                self.datas[data_name]['image'] = glob.glob(os.path.join(data, '*.jpg'))
                self.datas[data_name]['image'].sort()  # Sort the image paths
        print(self.datas.keys())  # Print the dataset keys for debugging


    def __getitem__(self, index):
        # Load image from 'input1'
        input1 = cv2.imread(self.datas['input1']['image'][index])
        input1 = input1.astype(dtype=np.float32)
        input1 = (input1 / 127.5) - 1.0  # Normalize to [-1, 1]
        input1 = np.transpose(input1, [2, 0, 1])  # Change channel order to (C, H, W)
        
        # Load image from 'input2'
        input2 = cv2.imread(self.datas['input2']['image'][index])
        input2 = input2.astype(dtype=np.float32)
        input2 = (input2 / 127.5) - 1.0  # Normalize to [-1, 1]
        input2 = np.transpose(input2, [2, 0, 1])  # Change channel order to (C, H, W)
        
        # Convert to tensor format for PyTorch
        input1_tensor = torch.tensor(input1)
        input2_tensor = torch.tensor(input2)

        return (input1_tensor, input2_tensor)  # Return the pair of images


    def __len__(self):
        # Return the total number of images in 'input1'
        return len(self.datas['input1']['image'])
