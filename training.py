import pathlib
import nibabel as nib
from monai.networks.nets import UNet
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import TrainDataset, EarlyStopping, TrainDatasetV2
from functions import *
import datetime
from monai.networks.layers.factories import Norm
import random

from torch.utils.tensorboard import SummaryWriter 


#SETTINGS
patch_size = (32, 32, 32)
stride = (16, 16, 16)
target_shape = (192, 224, 192) 

spatial_dims=3
in_channels=2
out_channels=1
net_channels = (16, 32, 64, 128, 256)
net_strides = (2, 2, 2, 2)
net_res_units = 2
norm=None

loss_fn = nn.MSELoss()
batch_size = 2
num_epochs = 50
note = "shallow unet"
augmentations = [2]
augmentation_dir = "axial"
timestamp = datetime.datetime.now().isoformat()
print(note)
print("Start at:", timestamp)

DATA_DIR = pathlib.Path("/proj/synthetic_alzheimer/users/x_almle/bobsrepository") 
LR_DIR = DATA_DIR / "LR_data" / "axial" / "even"
writer = SummaryWriter(log_dir= DATA_DIR / "tensorboard_logs" / timestamp)

assert LR_DIR.exists(), f"LR_DIR not found: {LR_DIR}"

#LOAD FILES AND SPLIT DATASET
t1_files = sorted(DATA_DIR.rglob("*T1w.nii.gz"))
t2_files = sorted(DATA_DIR.rglob("*T2w.nii.gz"))
t2_LR2_files = sorted(LR_DIR.rglob("*T2w_LR.nii.gz"))

assert len(t1_files) == len(t2_files) == len(t2_LR2_files), "Mismatch in number of files."

files = list(zip(t1_files, t2_files, t2_LR2_files))
train, val, test = split_dataset(files)


print(f"T1 files: {len(t1_files)}, T2 files: {len(t2_files)},LR2 files: {len(t2_LR2_files)}")
print(f"Train size: {len(train)}, Val size: {len(val)}, Test size: {len(test)}")

assert len(train) > 0 and len(val) > 0 and len(test) > 0, "One of the dataset splits is empty."

#SHUFFLE DATA
#random.shuffle(train)
#random.shuffle(val)
#random.shuffle(test)

# GPU/CPU detection
import os
slurm_gpus = int(os.environ.get('SLURM_GPUS_ON_NODE', '0'))
has_gpu = torch.cuda.is_available() and slurm_gpus > 0 and torch.cuda.device_count() > 0
device = torch.device("cuda" if has_gpu else "cpu")
cuda_version = torch.version.cuda
print(f"Using: {device} version {cuda_version} (SLURM GPUs: {slurm_gpus})")

#EXTRACT PATCHES
train_t1, train_t2, train_t2_LR = get_patches(train, patch_size, stride, target_shape)
val_t1, val_t2, val_t2_LR = get_patches(val, patch_size, stride, target_shape)
test_t1, test_t2, test_t2_LR = get_patches(test, patch_size, stride, target_shape)
print(f"Train patches: {len(train_t1)}, Val patches: {len(val_t1)}, Test patches: {len(test_t1)}")

#NETWORK TRAINING
train_dataset = TrainDataset(train_t1, train_t2_LR, train_t2)
#train_dataset = TrainDatasetV2(train_t1, train_t2)
train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
val_loader = DataLoader(TrainDataset(val_t1, val_t2_LR, val_t2), batch_size, shuffle=True)
#val_loader = DataLoader(TrainDatasetV2(val_t1, val_t2), batch_size, shuffle=True)

print(f"Number of training batches: {len(train_loader)}")

#INITIALIZE NETWORK
net = UNet(
    spatial_dims=spatial_dims,
    in_channels=in_channels,
    out_channels=out_channels,
    channels=net_channels,
    strides=net_strides,
    num_res_units=net_res_units,
    norm=norm,
).to(device, dtype=torch.float32)
loss_list = []
val_loss_list = []
optimizer = optim.Adam(net.parameters(), lr=1e-4)
print("Network initialized")

best_val_loss = float('inf')
early_stopping = EarlyStopping(patience=10, min_delta=0.0)

for epoch in range(num_epochs):
    #TRAINING
    net.train()
    train_loss = 0.0
    for batch in train_loader:
        input1, input2, target = batch
        inputs = torch.stack([input1, input2], dim=1).to(device, dtype=torch.float32, non_blocking=True)
        target = target.unsqueeze(1).to(device, dtype=torch.float32, non_blocking=True)
        #input, target = batch
        #inputs = input.unsqueeze(1).to(device, dtype=torch.float32, non_blocking=True)
        #target = target.unsqueeze(1).to(device, dtype=torch.float32, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = net(inputs)
        loss = loss_fn(outputs, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * inputs.size(0)

    #VALIDATION
    net.eval()
    with torch.no_grad():
        val_loss = 0.0
        for batch in val_loader:
            input1, input2, target = batch
            inputs = torch.stack([input1, input2], dim=1).to(device, dtype=torch.float32, non_blocking=True)
            target = target.unsqueeze(1).to(device, dtype=torch.float32, non_blocking=True) 
            #input, target = batch
            #inputs = input.unsqueeze(1).to(device, dtype=torch.float32, non_blocking=True)
            #target = target.unsqueeze(1).to(device, dtype=torch.float32, non_blocking=True)

            outputs = net(inputs)
            loss = loss_fn(outputs, target)
            val_loss += loss.item() * inputs.size(0)

    epoch_train_loss = train_loss / len(train_loader.dataset)
    loss_list.append(epoch_train_loss)
    epoch_val_loss = val_loss / len(val_loader.dataset)
    val_loss_list.append(epoch_val_loss)

    # Log epoch averages to TensorBoard
    writer.add_scalar('Training Loss', epoch_train_loss, epoch)
    writer.add_scalar('Validation Loss', epoch_val_loss, epoch)

    #SAVE BEST MODEL
    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        torch.save(net.state_dict(), DATA_DIR / "outputs" / f"{timestamp}_model_weights.pth")
        best_epoch = epoch + 1
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")

    #EARLY STOPPING
    if early_stopping.step(val_loss): #change to epoch_val_loss if needed?
        print(f"Early stopping at epoch {epoch+1}")
        break

#TESTING
generated_images = []
real_images = []

# LOAD BEST MODEL FOR TESTING
net = UNet(
    spatial_dims=spatial_dims,
    in_channels=in_channels,
    out_channels=out_channels,
    channels=net_channels,
    strides=net_strides,
    num_res_units=net_res_units,
    norm=norm,
)

net.load_state_dict(torch.load(DATA_DIR / "outputs" / f"{timestamp}_model_weights.pth", map_location=device))
net.to(device, dtype=torch.float32)
net.eval()
with torch.no_grad():
    for i in range(len(test_t1)):
        all_outputs = []
        for j in range(len(test_t2_LR[0])):
            input1 = torch.tensor(test_t1[i][j]).float()
            input2 = torch.tensor(test_t2_LR[i][j]).float()
            inputs = torch.stack([input1, input2], dim=0).unsqueeze(0) 
            inputs = inputs.to(device, dtype=torch.float32)
            #input = torch.tensor(test_t1[i][j]).float()
            #inputs = input.unsqueeze(0).unsqueeze(0).to(device, dtype=torch.float32)
            output = net(inputs)
            all_outputs.append(output.squeeze(0).squeeze(0).cpu().numpy())
        gen_reconstructed = reconstruct_from_patches(all_outputs, target_shape, stride)
        real_reconstructed = reconstruct_from_patches(test_t2[i], target_shape, stride)
        generated_images.append(gen_reconstructed)
        real_images.append(real_reconstructed)
        print(f"Processed test image {i+1}/{len(test_t1)}")

metrics = calculate_metrics(generated_images, real_images)

# SAVE RESULTS IN CSV

row_dict = {
    "note": note,
    "weights": f"{timestamp}_model_weights.pth",
    "start time": timestamp,
    "end time": datetime.datetime.now().isoformat(),
    "train_size": len(train),
    "val_size": len(val),
    "test_size": len(test),
    "anisotropic direction": augmentation_dir,
    "augmentations": str(augmentations),
    "patch_size": patch_size,
    "stride": stride,
    "target_shape": target_shape,
    "normalization": "min-max",
    "model": "MONAI 3D U-Net",
    "net spatial_dims": spatial_dims,
    "net in_channels": in_channels,
    "net out_channels": out_channels,
    "net channels": net_channels,
    "net strides": net_strides,
    "net num_res_units": net_res_units,
    "loss function": "MSELoss",
    "net norm": norm,
    "max num of epochs": num_epochs,
    "best_epoch": best_epoch,
    "batch_size": batch_size,
    "optimizer": "Adam",
    "learning_rate": optimizer.param_groups[0]['lr'],
    "early stopping patience": early_stopping.patience,
    "early stopping min_delta": early_stopping.min_delta,
    "psnr": metrics["psnr"], 
    "ssim": metrics["ssim"],
    "nrmse": metrics["nrmse"],
    "mse": metrics["mse"],
    "loss_list": loss_list,
    "val_loss_list": val_loss_list,
}

#create outputs directory if it doesn't exist
(DATA_DIR / "outputs").mkdir(parents=True, exist_ok=True)
append_row(DATA_DIR / "outputs" / "results2.csv", row_dict)
print("End at:", datetime.datetime.now().isoformat())

# Close TensorBoard writer
writer.close()
