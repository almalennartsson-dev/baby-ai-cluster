import pathlib
import nibabel as nib
import torch
import torch.nn as nn
from monai.networks.nets import UNet
from torch.utils.data import DataLoader
import torch.optim as optim
from generative.networks.nets import PatchDiscriminator
from generative.losses import PatchAdversarialLoss
from dataset import TrainDataset
from preprocessing import split_dataset, get_patches
import datetime
from file_structure import append_row
print("Start at:", datetime.datetime.now().isoformat())
# Parameters
batch_size = 2
patch_size = (32, 32, 32)
stride = (16, 16, 16)
target_shape = (192, 224, 192) 
num_epochs = 20
timestamp = datetime.datetime.now().isoformat()
lambda_adv = 0.01  # Weight for adversarial loss

# Smart GPU/CPU detection
import os
slurm_gpus = int(os.environ.get('SLURM_GPUS_ON_NODE', '0'))
has_gpu = torch.cuda.is_available() and slurm_gpus > 0 and torch.cuda.device_count() > 0

device = torch.device("cuda" if has_gpu else "cpu")
print(f"Using: {device} (SLURM GPUs: {slurm_gpus})")

# Define Generator and Discriminator
G = UNet(
    spatial_dims=3,
    in_channels=2,
    out_channels=1,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
    norm=None,
)

D = PatchDiscriminator(
    spatial_dims=3,
    num_channels=32,
    in_channels=3,
    out_channels=1,
)

# Define Loss Functions and Optimizers
adv_loss = PatchAdversarialLoss() #andra parametrar?
pix_loss = nn.L1Loss()

g_optimizer = optim.Adam(G.parameters(), lr=1e-4, betas=(0.5, 0.999))
d_optimizer = optim.Adam(D.parameters(), lr=1e-4, betas=(0.5, 0.999))

# Load data - make function of this?

#DATA_DIR = pathlib.Path.home()/"data"/"bobsrepository" #cluster?
DATA_DIR = pathlib.Path("/proj/synthetic_alzheimer/users/x_almle/bobsrepository") #cluster?
assert DATA_DIR.exists(), f"DATA_DIR not found: {DATA_DIR}"
t1_files = sorted(DATA_DIR.rglob("*T1w.nii.gz"))
t2_files = sorted(DATA_DIR.rglob("*T2w.nii.gz"))
t2_LR_files = sorted(DATA_DIR.rglob("*T2w_LR.nii.gz"))
ref_img = nib.load(str(t1_files[0]))
files = list(zip(t1_files, t2_files, t2_LR_files))
train, val, test = split_dataset(files)
train_t1, train_t2, train_t2_LR = get_patches(train, patch_size, stride, target_shape, ref_img)
val_t1, val_t2, val_t2_LR = get_patches(val, patch_size, stride, target_shape, ref_img)
test_t1, test_t2, test_t2_LR = get_patches(test, patch_size, stride, target_shape, ref_img)

# Define dataloaders
train_dataset = TrainDataset(train_t1, train_t2_LR, train_t2)
train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
val_loader = DataLoader(TrainDataset(val_t1, val_t2_LR, val_t2), batch_size, shuffle=True)


G.to(device, dtype=torch.float32)
D.to(device, dtype=torch.float32)

for epoch in range(num_epochs):
    G.train()
    D.train()

    for batch in train_loader:
        input1, input2, target = batch

        
        inputs = torch.stack([input1, input2], dim=1).to(device, dtype=torch.float32, non_blocking=True)  # (B, 2, 32, 32, 32)
        target = target.unsqueeze(1).to(device, dtype=torch.float32, non_blocking=True)  # (B, 1, 32, 32, 32)  
        
        #Generate fake image
        fake_output = G(inputs)

        real_pair = torch.cat([inputs, target], dim=1)  # (B, 3, 32, 32, 32)
        fake_pair = torch.cat([inputs, fake_output.detach()], dim=1)  # (B, 3, 32, 32, 32)
        
        #DISCRIMINATOR TRAINING
        d_optimizer.zero_grad()
        pred_real = D(real_pair)
        loss_real = adv_loss(pred_real[-1], target_is_real=True, for_discriminator=True)

        pred_fake = D(fake_pair)
        loss_fake = adv_loss(pred_fake[-1], target_is_real=False, for_discriminator=True)

        #Total loss
        loss_D = (loss_real + loss_fake) * 0.5
        loss_D.backward()
        d_optimizer.step()


        #GENERATOR TRAINING

        fake_pair = torch.cat([inputs, fake_output], dim=1)  # (B, 3, 32, 32, 32)
        pred_fake = D(fake_pair)
        
        g_optimizer.zero_grad()

        g_adv = adv_loss(pred_fake[-1], target_is_real=True, for_discriminator=False) #förstår inte det här steget
        g_pix = pix_loss(fake_output, target)
        
        #Total loss
        loss_G = g_pix + lambda_adv * g_adv
        loss_G.backward()
        g_optimizer.step()

        

    print(f"Epoch {epoch+1}/{num_epochs}, Generator Loss: {loss_G.item():.4f}, Discriminator Loss: {loss_D.item():.4f}")

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
    "net norm": "None",
    "num_epochs": num_epochs,
    "batch_size": batch_size,
    "learning_rate": g_optimizer.param_groups[0]['lr'],
    "pix_loss": "L1Loss",
    "adv_loss": "PatchAdversarialLoss",
    "optimizer": "Adam",
    "notes": "second gan try",
    "weights": f"GAN_{timestamp}_model_weights.pth",
}

#create outputs directory if it doesn't exist
(DATA_DIR / "outputs"/ "GAN" ).mkdir(parents=True, exist_ok=True)
torch.save(G.state_dict(), DATA_DIR / "outputs" / "GAN" / f"GAN_{timestamp}_model_weights.pth")
append_row(DATA_DIR / "outputs" / "GAN" / "GANresults.csv", row_dict)
print("End at:", datetime.datetime.now().isoformat())