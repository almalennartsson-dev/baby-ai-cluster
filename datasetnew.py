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
from functions import *

#want to input list of triplets of file paths
class TrainDataset(Dataset):
    def __init__(self, triplets, patch_size, stride, target_shape): 
        self.triplets = triplets
        self.patch_size = patch_size
        self.stride = stride
        self.target_shape = target_shape

        px = (self.target_shape[0] - self.patch_size[0]) // self.stride[0] + 1
        py = (self.target_shape[1] - self.patch_size[1]) // self.stride[1] + 1
        pz = (self.target_shape[2] - self.patch_size[2]) // self.stride[2] + 1
        self.patches_per_image = px * py * pz

        #cache patches for all images to speed up training
        self._cache_img_idx = None
        self._cache_patches = None


    def __len__(self):
        #total number of samples = images x patches per image
        return len(self.triplets) * self.patches_per_image

    def __getitem__(self, idx):

        #determine which image and which patch
        img_idx = idx // self.patches_per_image
        patch_idx = idx % self.patches_per_image

        # compute patches only when we move to a new image
        if self._cache_img_idx != img_idx:

            self._cache_patches = get_patches_from_triplet(self.triplets[img_idx], self.patch_size, self.stride, self.target_shape)
            self._cache_img_idx = img_idx

        t1_patches, t2_patches, t2lr_patches = self._cache_patches

        input1 = t1_patches[patch_idx]
        input2 = t2lr_patches[patch_idx]
        output = t2_patches[patch_idx]

        input1 = torch.from_numpy(input1).float()
        input2 = torch.from_numpy(input2).float()
        output = torch.from_numpy(output).float()

        
        #print(f"Returning sample {idx}: image {img_idx}, patch {patch_idx}")
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