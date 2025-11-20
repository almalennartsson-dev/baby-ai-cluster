import pathlib
import nibabel as nib
from monai.networks.nets import UNet
#from unet import UNet
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import TrainDataset, EarlyStopping
from preprocessing import create_and_save_LR_imgs, reconstruct_from_patches, split_dataset, get_patches
from file_structure import append_row
import datetime
from evaluations import calculate_metrics, RMSLELoss
from monai.networks.layers.factories import Norm
from monai.losses.perceptual import PerceptualLoss
import random

print("Start at:", datetime.datetime.now().isoformat())
#Collect all data files
#DATA_DIR = pathlib.Path.home()/"data"/"bobsrepository" #cluster?
DATA_DIR = pathlib.Path("/proj/synthetic_alzheimer/users/x_almle/bobsrepository") #cluster?
assert DATA_DIR.exists(), f"DATA_DIR not found: {DATA_DIR}"
t1_files = sorted(DATA_DIR.rglob("*T1w.nii.gz"))
t2_files = sorted(DATA_DIR.rglob("*T2w.nii.gz"))
t2_LR_files = sorted(DATA_DIR.rglob("*T2w_LR.nii.gz"))
t2_LR4_files = sorted(DATA_DIR.rglob("*T2w_LR4.nii.gz"))
files = list(zip(t1_files, t2_files, t2_LR_files))
files4 = list(zip(t1_files, t2_files, t2_LR4_files))
train, val, test = split_dataset(files)
train4, val4, test4 = split_dataset(files4)

train = train + train4
val = val + val4
test = test + test4

print(f"T1 files: {len(t1_files)}, T2 files: {len(t2_files)}, T2 LR files: {len(t2_LR_files)}, T2 LR4 files: {len(t2_LR4_files)}")
print(f"Train size: {len(train)}, Val size: {len(val)}, Test size: {len(test)}")

#SHUFFLE DATA
random.shuffle(train)
random.shuffle(val)
random.shuffle(test)


# Smart GPU/CPU detection
import os
slurm_gpus = int(os.environ.get('SLURM_GPUS_ON_NODE', '0'))
has_gpu = torch.cuda.is_available() and slurm_gpus > 0 and torch.cuda.device_count() > 0

device = torch.device("cuda" if has_gpu else "cpu")
print(f"Using: {device} (SLURM GPUs: {slurm_gpus})")

print("Starting training...")

#EXTRACT PATCHES

patch_size = (32, 32, 32)
stride = (16, 16, 16)
ref_img = nib.load(str(t1_files[0]))
target_shape = (192, 224, 192) 
train_t1, train_t2, train_t2_LR = get_patches(train, patch_size, stride, target_shape, ref_img)
val_t1, val_t2, val_t2_LR = get_patches(val, patch_size, stride, target_shape, ref_img)
test_t1, test_t2, test_t2_LR = get_patches(test, patch_size, stride, target_shape, ref_img)

print(f"Train patches: {len(train_t1)}, Val patches: {len(val_t1)}, Test patches: {len(test_t1)}")

#NETWORK TRAINING
batch_size = 2

train_dataset = TrainDataset(train_t1, train_t2_LR, train_t2)
train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
val_loader = DataLoader(TrainDataset(val_t1, val_t2_LR, val_t2), batch_size, shuffle=True)

print(f"Number of training batches: {len(train_loader)}")
net = UNet(
    spatial_dims=3,
    in_channels=2,
    out_channels=1,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
    norm=None,
)
net.to(device, dtype=torch.float32)
print("Network initialized")
loss_fn = nn.MSELoss()
lpips_loss = PerceptualLoss(
   spatial_dims=3,
    network_type='medicalnet_resnet10_23datasets',
    is_fake_3d=False,
).to(device, dtype=torch.float32)
w_lpips = 0.1
print("Loss functions initialized")
loss_list = []
val_loss_list = []
optimizer = optim.Adam(net.parameters(), lr=1e-4)
num_epochs = 50
print(f"Number of epochs: {num_epochs}")

#use_cuda = torch.cuda.is_available()
#print(f"Using CUDA: {use_cuda}")
#device = torch.device("cuda" if use_cuda else "cpu")
#device = torch.device("cpu") #cluster?



#lpips_loss = lpips_loss.to(device=device, dtype=torch.float32)
#lpips_loss.eval()
timestamp = datetime.datetime.now().isoformat()
best_val_loss = float('inf')
early_stopping = EarlyStopping(patience=5, min_delta=0.0)

for epoch in range(num_epochs):
    #TRAINING
    net.train()
    train_loss = 0.0
    for batch in train_loader:
        input1, input2, target = batch
        inputs = torch.stack([input1, input2], dim=1).to(device, dtype=torch.float32, non_blocking=True)  # (B, 2, 64, 64, 64)
        target = target.unsqueeze(1).to(device, dtype=torch.float32, non_blocking=True)  # (B, 1, 64, 64, 64)

        optimizer.zero_grad(set_to_none=True)
        outputs = net(inputs)
        pix_loss = loss_fn(outputs, target)
        perc_loss = lpips_loss(outputs, target)
        loss = pix_loss + w_lpips * perc_loss
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * inputs.size(0)
       

    #VALIDATION
    net.eval()
    with torch.no_grad():
        val_loss = 0.0
        for batch in val_loader:
            input1, input2, target = batch
            inputs = torch.stack([input1, input2], dim=1).to(device, dtype=torch.float32, non_blocking=True)  # (B, 2, 64, 64, 64)
            target = target.unsqueeze(1).to(device, dtype=torch.float32, non_blocking=True)  # (B, 1, 64, 64, 64)

            outputs = net(inputs)
            loss = loss_fn(outputs, target)
            val_loss += loss.item() * inputs.size(0)

    epoch_train_loss = train_loss / len(train_loader.dataset)
    loss_list.append(epoch_train_loss)
    epoch_val_loss = val_loss / len(val_loader.dataset)
    val_loss_list.append(epoch_val_loss)

    #save the best model based on validation loss. KEEP THIS OR NOT?
    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        torch.save(net.state_dict(), DATA_DIR / "outputs" / f"{timestamp}_model_weights.pth")
        best_epoch = epoch + 1 # Store the best epoch number

    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")

    #EARLY STOPPING
    if early_stopping.step(val_loss):
        print(f"Early stopping at epoch {epoch+1}")
        break

#TESTING
generated_images = []
real_images = []

# Load the best model for testing
net = UNet(
    spatial_dims=3,
    in_channels=2,
    out_channels=1,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
    norm=None,
)

net.load_state_dict(torch.load(DATA_DIR / "outputs" / f"{timestamp}_model_weights.pth", map_location=device))
net.to(device, dtype=torch.float32)
net.eval() # could be not the best model, but the last one. FIX THIS!
with torch.no_grad():
    for i in range(len(test_t1)):
        all_outputs = []
        for j in range(len(test_t1[0])):
            input1 = torch.tensor(test_t1[i][j]).float()
            input2 = torch.tensor(test_t2_LR[i][j]).float()
            inputs = torch.stack([input1, input2], dim=0).unsqueeze(0)  # (1, 2, 16, 16, 16)
            inputs = inputs.to(device, dtype=torch.float32)  # Move to device!
            output = net(inputs)
            all_outputs.append(output.squeeze(0).squeeze(0).cpu().numpy())  # (64, 64, 64)
        gen_reconstructed = reconstruct_from_patches(all_outputs, target_shape, stride)
        real_reconstructed = reconstruct_from_patches(test_t2[i], target_shape, stride)
        generated_images.append(gen_reconstructed)
        real_images.append(real_reconstructed)
        print(f"Processed test image {i+1}/{len(test_t1)}")

metrics = calculate_metrics(generated_images, real_images)

# SAVE RESULTS

row_dict = {
    "timestamp": timestamp,
    "train_size": len(train),
    "val_size": len(val),
    "test_size": len(test),
    "patch_size": patch_size,
    "stride": stride,
    "target_shape": target_shape,
    "normalization": "min-max",
    "model": "MONAI 3D U-Net",
    "net spatial_dims": 3,
    "net in_channels": 2,
    "net out_channels": 1,
    "net channels": (16, 32, 64, 128, 256),
    "net strides": (2, 2, 2, 2),
    "net num_res_units": 2,
    "net norm": None,
    "num_epochs": num_epochs,
    "batch_size": batch_size,
    "learning_rate": optimizer.param_groups[0]['lr'],
    "psnr": metrics["psnr"], 
    "ssim": metrics["ssim"],
    "lpips": None,
    "nrmse": metrics["nrmse"],
    "mse": metrics["mse"],
    "loss_fn": "MSLELoss",
    "loss_list": loss_list,
    "optimizer": "Adam",
    "masking": "None",
    "weights": f"{timestamp}_model_weights.pth",
    "val_loss_list": val_loss_list,
    "best_epoch": best_epoch,
    "stop_epoch": epoch + 1,
    "patience": early_stopping.patience,
    "min_delta": early_stopping.min_delta,
    "notes":"try with LR4 augementation",
}

#create outputs directory if it doesn't exist
(DATA_DIR / "outputs").mkdir(parents=True, exist_ok=True)
#torch.save(net.state_dict(), DATA_DIR / "outputs" / f"{timestamp}_model_weights.pth") 
append_row(DATA_DIR / "outputs" / "results.csv", row_dict)
print("End at:", datetime.datetime.now().isoformat())
