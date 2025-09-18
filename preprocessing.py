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
    def __init__(self, csv_file, img_dir): # transform=None, target_transform=None):
        self.img_labels = pd.read_csv(csv_file) #csv file innehåller img name flr hr lr
        self.img_dir = img_dir


    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        HR_img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        LR_img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 1])

        HR_img = nib.load(str(HR_img_path)).get_fdata()
        LR_img = nib.load(str(LR_img_path)).get_fdata()

        # Optionally, convert to torch tensors
        HR_img = torch.from_numpy(HR_img).float()
        LR_img = torch.from_numpy(LR_img).float()

        return HR_img, LR_img
    




