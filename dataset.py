import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import os
import pandas as pd
from torchvision.io import decode_image
import nibabel as nib
import matplotlib.pyplot as plt


class TrainDataset(Dataset):
    def __init__(self, input1, input2, output): 
        self.input1 = [patch for img_patches in input1 for patch in img_patches]
        self.input2 = [patch for img_patches in input2 for patch in img_patches]
        self.output = [patch for img_patches in output for patch in img_patches]

    def __len__(self):
        return len(self.output)

    def __getitem__(self, idx):
        input1 = self.input1[idx]
        input2 = self.input2[idx]
        output = self.output[idx]

        input1 = torch.from_numpy(input1).float()
        input2 = torch.from_numpy(input2).float()
        output = torch.from_numpy(output).float()

        return input1, input2, output

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best = None
        self.count = 0
        self.should_stop = False

    def step(self, metric):
        if self.best is None: #first loop
            self.best = metric
            return False

        improve = (metric < self.best - self.min_delta) 
        if improve:
            self.best = metric
            self.count = 0
        else:
            self.count += 1
            if self.count >= self.patience:
                self.should_stop = True
        return self.should_stop