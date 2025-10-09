import pathlib
import nibabel as nib
from monai.networks.nets import UNet
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import TrainDataset
from preprocessing import create_and_save_LR_imgs, reconstruct_from_patches, split_dataset, get_patches
from file_structure import append_row
import datetime
from evaluations import calculate_metrics 

print("Start at:", datetime.datetime.now().isoformat())
#Collect all data files
DATA_DIR = pathlib.Path.home()/"data"/"bobsrepository" #cluster?
#DATA_DIR = pathlib.Path("/proj/synthetic_alzheimer/users/x_almle/bobsrepository") #cluster?
assert DATA_DIR.exists(), f"DATA_DIR not found: {DATA_DIR}"
t1_files = sorted(DATA_DIR.rglob("*T1w.nii.gz"))
t2_files = sorted(DATA_DIR.rglob("*T2w.nii.gz"))
t2_LR_files = sorted(DATA_DIR.rglob("*T2w_LR.nii.gz"))
files = list(zip(t1_files, t2_files, t2_LR_files))

print(f"T1 files: {len(t1_files)}, T2 files: {len(t2_files)}, T2 LR files: {len(t2_LR_files)}")

#SPLIT DATASET
train, val, test = split_dataset(files)


#EXTRACT PATCHES

patch_size = (16, 16, 16)
stride = (8, 8, 8)
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
print("Network initialized")
loss_fn = nn.MSELoss()
loss_list = []
optimizer = optim.Adam(net.parameters(), lr=1e-4)
num_epochs = 100
print(f"Number of epochs: {num_epochs}")

#use_cuda = torch.cuda.is_available()
#print(f"Using CUDA: {use_cuda}")
#device = torch.device("cuda" if use_cuda else "cpu")
#device = torch.device("cpu") #cluster?

# Smart GPU/CPU detection
import os
slurm_gpus = int(os.environ.get('SLURM_GPUS_ON_NODE', '0'))
has_gpu = torch.cuda.is_available() and slurm_gpus > 0 and torch.cuda.device_count() > 0

device = torch.device("cuda" if has_gpu else "cpu")
print(f"Using: {device} (SLURM GPUs: {slurm_gpus})")

net.to(device, dtype=torch.float32)

print("Starting training...")
for epoch in range(num_epochs):
    net.train()
    running_loss = 0.0
    for batch in train_loader:
        input1, input2, target = batch
        inputs = torch.stack([input1, input2], dim=1).to(device, dtype=torch.float32, non_blocking=True)  # (B, 2, 64, 64, 64)
        target = target.unsqueeze(1).to(device, dtype=torch.float32, non_blocking=True)  # (B, 1, 64, 64, 64)

        optimizer.zero_grad(set_to_none=True)
        outputs = net(inputs)
        loss = loss_fn(outputs, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    loss_list.append(epoch_loss)

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

#VALIDATION
generated_images = []
real_images = []

net.eval()
with torch.no_grad():
    for i in range(len(val_t1)):
        all_outputs = []
        for j in range(len(val_t1[0])):
            input1 = torch.tensor(val_t1[i][j]).float()
            input2 = torch.tensor(val_t2_LR[i][j]).float()
            inputs = torch.stack([input1, input2], dim=0).unsqueeze(0)  # (1, 2, 64, 64, 64)
            inputs = inputs.to(device, dtype=torch.float32)  # Move to device!
            output = net(inputs)
            all_outputs.append(output.squeeze(0).squeeze(0).cpu().numpy())  # (64, 64, 64)
        gen_reconstructed = reconstruct_from_patches(all_outputs, target_shape, stride)
        real_reconstructed = reconstruct_from_patches(val_t2[i], target_shape, stride)
        generated_images.append(gen_reconstructed)
        real_images.append(real_reconstructed)
        print(f"Processed validation image {i+1}/{len(val_t1)}")

metrics = calculate_metrics(generated_images, real_images)

# SAVE RESULTS
timestamp = datetime.datetime.now().isoformat()

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
    "loss_fn": "MSELoss",
    "loss_list": loss_list,
    "optimizer": "Adam",
    "notes": "introduce correct split",
    "masking": "None",
    "weights": f"{timestamp}_model_weights.pth",

}

#create outputs directory if it doesn't exist
(DATA_DIR / "outputs").mkdir(parents=True, exist_ok=True)
torch.save(net.state_dict(), DATA_DIR / "outputs" / f"{timestamp}_model_weights.pth") 
append_row(DATA_DIR / "outputs" / "results.csv", row_dict)
print("End at:", datetime.datetime.now().isoformat())
