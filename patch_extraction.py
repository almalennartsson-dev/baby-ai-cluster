import pathlib
import nibabel as nib
from monai.networks.nets import UNet
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import TrainDataset, EarlyStopping
from functions import *
import datetime
from monai.networks.layers.factories import Norm
from monai.losses.perceptual import PerceptualLoss
import random
from torch.utils.tensorboard import SummaryWriter 

print("script starting...")

#SETTINGS
patch_size = (32, 32, 32)
stride = (16, 16, 16)
target_shape = (192, 224, 192)
augmentations = [2,3,4,5,8]
augmentation_dir = "all_directions" 

DATA_DIR = pathlib.Path("/proj/synthetic_alzheimer/users/x_almle/bobsrepository") 

AX_DIR = DATA_DIR / "LR_data" / "axial" / "even"
CO_DIR = DATA_DIR / "LR_data" / "coronal" / "even"
SA_DIR = DATA_DIR / "LR_data" / "sagittal" / "even"

assert AX_DIR.exists(), f"AX_DIR not found: {AX_DIR}"
assert CO_DIR.exists(), f"CO_DIR not found: {CO_DIR}"
assert SA_DIR.exists(), f"SA_DIR not found: {SA_DIR}"

#load GT files
t1_files = sorted(DATA_DIR.rglob("*T1w.nii.gz"))
t2_files = sorted(DATA_DIR.rglob("*T2w.nii.gz"))

print(f"T1 files: {len(t1_files)}, T2 files: {len(t2_files)}")
#AXIAL
#load all files
t2_ax_LR2_files = sorted(AX_DIR.rglob("*T2w_LR.nii.gz"))
t2_ax_LR3_files = sorted(AX_DIR.rglob("*T2w_LR3.nii.gz"))
t2_ax_LR4_files = sorted(AX_DIR.rglob("*T2w_LR4.nii.gz"))
t2_ax_LR5_files = sorted(AX_DIR.rglob("*T2w_LR5.nii.gz"))
t2_ax_LR8_files = sorted(AX_DIR.rglob("*T2w_LR8.nii.gz"))
print(f"Axial LR2 files: {len(t2_ax_LR2_files)}, LR3 files: {len(t2_ax_LR3_files)}, LR4 files: {len(t2_ax_LR4_files)}, LR5 files: {len(t2_ax_LR5_files)}, LR8 files: {len(t2_ax_LR8_files)}")
#combine all LR files
files_ax_LR2 = list(zip(t1_files, t2_files, t2_ax_LR2_files))
files_ax_LR3 = list(zip(t1_files, t2_files, t2_ax_LR3_files))
files_ax_LR4 = list(zip(t1_files, t2_files, t2_ax_LR4_files))
files_ax_LR5 = list(zip(t1_files, t2_files, t2_ax_LR5_files))
files_ax_LR8 = list(zip(t1_files, t2_files, t2_ax_LR8_files))
#split datasets
train_ax_LR2, val_ax_LR2, test_ax_LR2 = split_dataset(files_ax_LR2)
train_ax_LR3, val_ax_LR3, test_ax_LR3 = split_dataset(files_ax_LR3)
train_ax_LR4, val_ax_LR4, test_ax_LR4 = split_dataset(files_ax_LR4)
train_ax_LR5, val_ax_LR5, test_ax_LR5 = split_dataset(files_ax_LR5)
train_ax_LR8, val_ax_LR8, test_ax_LR8 = split_dataset(files_ax_LR8)

#CORONAL
#load all files
t2_co_LR2_files = sorted(CO_DIR.rglob("*T2w_LR2.nii.gz"))
t2_co_LR3_files = sorted(CO_DIR.rglob("*T2w_LR3.nii.gz"))
t2_co_LR4_files = sorted(CO_DIR.rglob("*T2w_LR4.nii.gz"))
t2_co_LR5_files = sorted(CO_DIR.rglob("*T2w_LR5.nii.gz"))
t2_co_LR8_files = sorted(CO_DIR.rglob("*T2w_LR8.nii.gz"))
print(f"Coronal LR2 files: {len(t2_co_LR2_files)}, LR3 files: {len(t2_co_LR3_files)}, LR4 files: {len(t2_co_LR4_files)}, LR5 files: {len(t2_co_LR5_files)}, LR8 files: {len(t2_co_LR8_files)}")
#combine all LR files
files_co_LR2 = list(zip(t1_files, t2_files, t2_co_LR2_files))
files_co_LR3 = list(zip(t1_files, t2_files, t2_co_LR3_files))
files_co_LR4 = list(zip(t1_files, t2_files, t2_co_LR4_files))
files_co_LR5 = list(zip(t1_files, t2_files, t2_co_LR5_files))
files_co_LR8 = list(zip(t1_files, t2_files, t2_co_LR8_files))
#split datasets
train_co_LR2, val_co_LR2, test_co_LR2 = split_dataset(files_co_LR2)
train_co_LR3, val_co_LR3, test_co_LR3 = split_dataset(files_co_LR3)
train_co_LR4, val_co_LR4, test_co_LR4 = split_dataset(files_co_LR4)
train_co_LR5, val_co_LR5, test_co_LR5 = split_dataset(files_co_LR5)
train_co_LR8, val_co_LR8, test_co_LR8 = split_dataset(files_co_LR8)

#SAGITTAL
#load all files
t2_sa_LR2_files = sorted(SA_DIR.rglob("*T2w_LR2.nii.gz"))
t2_sa_LR3_files = sorted(SA_DIR.rglob("*T2w_LR3.nii.gz"))
t2_sa_LR4_files = sorted(SA_DIR.rglob("*T2w_LR4.nii.gz"))
t2_sa_LR5_files = sorted(SA_DIR.rglob("*T2w_LR5.nii.gz"))
t2_sa_LR8_files = sorted(SA_DIR.rglob("*T2w_LR8.nii.gz"))
print(f"Sagittal LR2 files: {len(t2_sa_LR2_files)}, LR3 files: {len(t2_sa_LR3_files)}, LR4 files: {len(t2_sa_LR4_files)}, LR5 files: {len(t2_sa_LR5_files)}, LR8 files: {len(t2_sa_LR8_files)}")
#combine all LR files
files_sa_LR2 = list(zip(t1_files, t2_files, t2_sa_LR2_files))
files_sa_LR3 = list(zip(t1_files, t2_files, t2_sa_LR3_files))
files_sa_LR4 = list(zip(t1_files, t2_files, t2_sa_LR4_files))
files_sa_LR5 = list(zip(t1_files, t2_files, t2_sa_LR5_files))
files_sa_LR8 = list(zip(t1_files, t2_files, t2_sa_LR8_files))
#split datasets
train_sa_LR2, val_sa_LR2, test_sa_LR2 = split_dataset(files_sa_LR2)
train_sa_LR3, val_sa_LR3, test_sa_LR3 = split_dataset(files_sa_LR3)
train_sa_LR4, val_sa_LR4, test_sa_LR4 = split_dataset(files_sa_LR4) 
train_sa_LR5, val_sa_LR5, test_sa_LR5 = split_dataset(files_sa_LR5)
train_sa_LR8, val_sa_LR8, test_sa_LR8 = split_dataset(files_sa_LR8)


#COMBINE ALL ORIENTATIONS
train = train_ax_LR2 + train_ax_LR3 + train_ax_LR4 + train_ax_LR5 + train_ax_LR8 + \
        train_co_LR2 + train_co_LR3 + train_co_LR4 + train_co_LR5 + train_co_LR8 + \
        train_sa_LR2 + train_sa_LR3 + train_sa_LR4 + train_sa_LR5 + train_sa_LR8
val = val_ax_LR2 + val_ax_LR3 + val_ax_LR4 + val_ax_LR5 + val_ax_LR8 + \
      val_co_LR2 + val_co_LR3 + val_co_LR4 + val_co_LR5 + val_co_LR8 + \
      val_sa_LR2 + val_sa_LR3 + val_sa_LR4 + val_sa_LR5 + val_sa_LR8
test = test_ax_LR2 + test_ax_LR3 + test_ax_LR4 + test_ax_LR5 + test_ax_LR8 + \
       test_co_LR2 + test_co_LR3 + test_co_LR4 + test_co_LR5 + test_co_LR8 + \
       test_sa_LR2 + test_sa_LR3 + test_sa_LR4 + test_sa_LR5 + test_sa_LR8  

#SHUFFLE DATA
random.shuffle(train)
random.shuffle(val)
random.shuffle(test)

print(f"Train size: {len(train)}, Val size: {len(val)}, Test size: {len(test)}")

#EXTRACT PATCHES

train_t1, train_t2, train_t2_LR = get_patches(train, patch_size, stride, target_shape)
val_t1, val_t2, val_t2_LR = get_patches(val, patch_size, stride, target_shape)
test_t1, test_t2, test_t2_LR = get_patches(test, patch_size, stride, target_shape)

print(f"Train patches: {len(train_t1)}, Val patches: {len(val_t1)}, Test patches: {len(test_t1)}")

torch.save(
    {
    "train_t1": train_t1,
    "train_t2": train_t2,
    "train_t2_LR": train_t2_LR,
    "val_t1": val_t1,
    "val_t2": val_t2,
    "val_t2_LR": val_t2_LR,
    "test_t1": test_t1,
    "test_t2": test_t2,
    "test_t2_LR": test_t2_LR,
    "meta": {
        "patch_size": patch_size,
        "stride": stride,
        "target_shape": target_shape,
    }
}, DATA_DIR / "patches.pt")

print("End at:", datetime.datetime.now().isoformat())
