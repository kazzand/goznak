import os
import torch
import numpy as np
import torch.nn as nn
from torchvision import models, transforms, datasets
from functions import spec2image

class CLFData(torch.utils.data.Dataset):
    def __init__(self, path_data):
        self.path_data = path_data
        self.class1 = os.listdir(self.path_data + 'noisy/')
        self.class1.sort()
        self.class2 = os.listdir(self.path_data + 'clear/')
        self.class2.sort()
        self.labels0 = np.zeros(len(self.class1))
        self.labels1 = np.ones(len(self.class2))
        self.all_paths_class1 = np.array([self.path_data + 'noisy/' + name for name in self.class1])
        self.all_paths_class2 = np.array([self.path_data + 'clear/' + name for name in self.class2])
        self.all_paths = np.hstack((self.all_paths_class1,self.all_paths_class2))
        self.labels = np.hstack((self.labels0, self.labels1))
    def __len__(self):
        return len(self.all_paths)
    def __getitem__(self, idx):
        image = spec2image(np.load(self.all_paths[idx]).T)
        if image.shape[1] <= 200:
            image_crop = np.zeros((80,200))
            image_crop[:image.shape[0],:image.shape[1]] = image
        else:
            rand_coord = np.random.randint(0, image.shape[1]-200)
            image_crop = image[:, rand_coord:rand_coord+200]
        image_crop = torch.FloatTensor(image_crop)
        image_crop = image_crop.unsqueeze(0)
        image_crop.requires_grad_(True)
        return image_crop, torch.FloatTensor([self.labels[idx]])


class DenoiseData(torch.utils.data.Dataset):
    def __init__(self, path_data):
        self.path_data = path_data
        self.files_data = os.listdir(self.path_data + 'noisy')
        self.files_data.sort()
        self.files_target = os.listdir(self.path_data + 'clear')
        self.files_target.sort()
        self.all_paths_data = [self.path_data + 'noisy/' + name for name in self.files_data]
        self.all_paths_target = [self.path_data + 'clear/' + name for name in self.files_target]
    def __len__(self):
        return len(self.files_data)
    def __getitem__(self, idx):
        data = spec2image(np.load(self.all_paths_data[idx]).T)
        target = spec2image(np.load(self.all_paths_target[idx]).T)
        if data.shape[1] <= 200:
            data_crop = np.zeros((80,200))
            target_crop = np.zeros((80,200))
            data_crop[:data.shape[0],:data.shape[1]] = data
            target_crop[:target.shape[0],:target.shape[1]] = target
        else:
            rand_coord = np.random.randint(0, data.shape[1]-200)
            data_crop = data[:, rand_coord:rand_coord+200]
            target_crop = target[:, rand_coord:rand_coord+200]
        data_crop = torch.FloatTensor(data_crop).unsqueeze(0)
        target_crop = torch.FloatTensor(target_crop).unsqueeze(0)
        data_crop.requires_grad_(True)
        target_crop.requires_grad_(True)
        return data_crop, target_crop