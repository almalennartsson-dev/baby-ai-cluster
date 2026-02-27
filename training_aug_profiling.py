import pathlib
import nibabel as nib
from monai.networks.nets import UNet
import torch
import torch.nn as nn
import torch.optim as optim
import torch.profiler
from torch.utils.data import DataLoader
from torch.profiler import profile, ProfilerActivity, record_function, schedule
from datasetnew import TrainDataset, EarlyStopping
from functions import *
import datetime
from monai.networks.layers.factories import Norm
from monai.losses.perceptual import PerceptualLoss
import random
from torch.utils.tensorboard import SummaryWriter 
import time

print("script starting...")

#SETTINGS
patch_size = (32, 32, 32)
stride = (16, 16, 16)
target_shape = (192, 224, 192)
augmentations = [2,3,4,5,8]
augmentation_dir = "all_directions" 

spatial_dims=3
in_channels=2
out_channels=1
net_channels = (32, 64, 128, 256, 512, 1024)
net_strides = (2, 2, 2, 2, 2)
net_res_units = 10
norm=None

loss_fn = nn.MSELoss()
batch_size = 2
num_epochs = 50
note = "Augmented in 3 directions, downsampled with 2,3,4,5,8"
timestamp = datetime.datetime.now().isoformat()

print(note)
print("Start at:", timestamp)

DATA_DIR = pathlib.Path("/proj/synthetic_alzheimer/users/x_almle/bobsrepository") 

writer = SummaryWriter(log_dir= DATA_DIR / "tensorboard_logs" / timestamp)

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

# GPU/CPU detection
import os
slurm_gpus = int(os.environ.get('SLURM_GPUS_ON_NODE', '0'))
has_gpu = torch.cuda.is_available() and slurm_gpus > 0 and torch.cuda.device_count() > 0
device = torch.device("cuda" if has_gpu else "cpu")
print(f"Using: {device} (SLURM GPUs: {slurm_gpus})")

#PROFILING

PROFILE = True
profile_epoch = 0
warmup_steps = 5
active_steps = 50

prof = None
if PROFILE:
    prof_dir = DATA_DIR / "tensorboard_logs" / timestamp / "profiler"
    prof_dir.mkdir(parents=True, exist_ok=True)

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    prof = profile(
        activities=activities,
        schedule=schedule(wait=0, warmup=warmup_steps, active=active_steps, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(str(prof_dir)),
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    )

print("Starting training...")

#NETWORK TRAINING
train_dataset = TrainDataset(train, patch_size, stride, target_shape)
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=4,  # or more, experiment
    pin_memory=(device.type == "cuda"),
    persistent_workers=True,
)
val_loader = DataLoader(
    TrainDataset(val, patch_size, stride, target_shape),
    batch_size=batch_size,
    shuffle=False,
    num_workers=4,
    pin_memory=(device.type == "cuda"),
    persistent_workers=True,
)
print(f"Number of training batches: {len(train_loader)}")
net = UNet(
    spatial_dims=spatial_dims,
    in_channels=in_channels,
    out_channels=out_channels,
    channels=net_channels,
    strides=net_strides,
    num_res_units=net_res_units,
    norm=norm,
)
net.to(device, dtype=torch.float32)
loss_list = []
val_loss_list = []
optimizer = optim.Adam(net.parameters(), lr=1e-4)
print("Network initialized")

best_val_loss = float('inf')
early_stopping = EarlyStopping(patience=5, min_delta=0.0)

for epoch in range(num_epochs):

    # ---- TRAINING ----
    net.train()
    train_loss = 0.0
    data_wait_times = []

    if PROFILE and epoch == profile_epoch:
        with prof:

            it = iter(train_loader)
            t_prev = time.perf_counter()
            #for step, batch in enumerate(train_loader):
            for step in range(warmup_steps + active_steps +1):
                t0 = time.perf_counter()
                with record_function("data_load"):
                    batch = next(it)
                t1 = time.perf_counter()
                data_wait_s = t1 - t0
                data_wait_times.append(data_wait_s)


                with record_function("data_wait"):
                    pass  # just to record the data waiting time
                
                with record_function("batch_unpack"):
                    input1, input2, target = batch

                with record_function("h2d_and_stack"):
                    inputs = torch.stack([input1, input2], dim=1).to(device, dtype=torch.float32, non_blocking=True)
                    target = target.unsqueeze(1).to(device, dtype=torch.float32, non_blocking=True)

                with record_function("zero_grad"):
                    optimizer.zero_grad(set_to_none=True)

                with record_function("forward"):
                    outputs = net(inputs)

                with record_function("loss"):
                    loss = loss_fn(outputs, target)

                with record_function("backward"):
                    loss.backward()

                with record_function("optimizer_step"):
                    optimizer.step()

                train_loss += loss.item() * inputs.size(0)

                prof.step()

                # stop after warmup+active completed (keeps trace small)
                if step >= warmup_steps + active_steps:
                    break

        print("Profiler trace written to:", prof_dir)
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
        if data_wait_times:
            # Ignore warmup steps
            measured = data_wait_times[warmup_steps:]

            avg_wait = sum(measured) / len(measured)
            max_wait = max(measured)
            min_wait = min(measured)

            print("\n=== DataLoader Wait Time ===")
            print(f"Avg wait per batch: {avg_wait*1000:.2f} ms")
            print(f"Max wait per batch: {max_wait*1000:.2f} ms")
            print(f"Min wait per batch: {min_wait*1000:.2f} ms")
        PROFILE = False  # profile only once

    else:
        # normal (unprofiled) training
        for step, batch in enumerate(train_loader):
            input1, input2, target = batch
            inputs = torch.stack([input1, input2], dim=1).to(device, dtype=torch.float32, non_blocking=True)
            target = target.unsqueeze(1).to(device, dtype=torch.float32, non_blocking=True)

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

            outputs = net(inputs)
            loss = loss_fn(outputs, target)
            val_loss += loss.item() * inputs.size(0)

    if PROFILE and epoch == profile_epoch:
        prof.stop()
        print("Profiler trace written to:", prof_dir)

        PROFILE = False  # do not profile later epochs
    
    epoch_train_loss = train_loss / len(train_loader.dataset)
    loss_list.append(epoch_train_loss)
    epoch_val_loss = val_loss / len(val_loader.dataset)
    val_loss_list.append(epoch_val_loss)

    writer.add_scalar('Loss/Train', epoch_train_loss, epoch)
    writer.add_scalar('Loss/Validation', epoch_val_loss, epoch)

    #save the best model based on validation loss
    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        torch.save(net.state_dict(), DATA_DIR / "outputs" / f"{timestamp}_model_weights.pth")
        best_epoch = epoch + 1 # Store the best epoch number

    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")

    #EARLY STOPPING
    if early_stopping.step(val_loss):
        print(f"Early stopping at epoch {epoch+1}")
        break

# SAVE RESULTS

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
    "psnr": "", 
    "ssim": "",
    "nrmse": "",
    "mse": "",
    "loss_list": loss_list,
    "val_loss_list": val_loss_list,
}

#create outputs directory if it doesn't exist
(DATA_DIR / "outputs").mkdir(parents=True, exist_ok=True)
append_row(DATA_DIR / "outputs" / "results2.csv", row_dict)
print("End at:", datetime.datetime.now().isoformat())
writer.close()