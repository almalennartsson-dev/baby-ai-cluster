import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import os
import pandas as pd
from torchvision.io import decode_image
import nibabel as nib
from nibabel import processing
import matplotlib.pyplot as plt
import scipy.ndimage
from scipy.ndimage import zoom
from skimage.metrics import structural_similarity, peak_signal_noise_ratio, mean_squared_error, normalized_root_mse


def split_dataset(file_list, train_ratio=(0.7, 0.15, 0.15)):
    """
    Splits the dataset into training and validation sets based on the given ratio.
    
    Parameters:
    - file_list: List of file paths.
    - train_ratio: Tuple indicating the ratio for train, val, and test splits. Default is (0.7, 0.15, 0.15).    
    Returns:
    - train_files: List of training file paths.
    - val_files: List of validation file paths.
    - test_files: List of test file paths.
    """
    train, val, test = train_ratio
    
    #random.shuffle(file_list)
    train_split = int(len(file_list) * train)
    val_split = int(len(file_list) * (train + val))
    train_files = file_list[:train_split]
    val_files = file_list[train_split:val_split]
    test_files = file_list[val_split:]
    return train_files, val_files, test_files

def create_LR_img(img, scale_factor):
    """
    Downsamples a 3D image by the given scale factor. Interpolates to match original shape.
    
    Parameters:
    img (numpy array): The input 3D image to be LR.
    scale_factor (int): The factor by which to downsample the image.
    
    Returns:
    numpy array: The LR 3D image.
    """

    down_img = img[:, :, ::scale_factor]
    
    # Calculate zoom factors to match original shape
    zoom_factors = np.array(img.shape) / np.array(down_img.shape)

    # Upsample with interpolation
    up_img = zoom(down_img, zoom_factors, order=3)  # order=3: cubic interpolation

    return up_img

def create_and_save_LR_imgs(file_list, scale_factor, output_dir):
    """
    Creates and saves low-resolution (LR) images from a list of high-resolution (HR) image files.

    Parameters:
    - file_list: List of file paths to the HR images.
    - scale_factor: The factor by which to downsample the images.
    - output_dir: Directory where the LR images will be saved.
    - base_name: Base name for the saved LR image files.
    Returns:
    list: List of file paths to the saved LR images.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    saved_paths = []
    for i, file_path in enumerate(file_list):
        img = nib.load(file_path)
        img_data = img.get_fdata()
        lr_img = create_LR_img(img_data, scale_factor)
        lr_img_nib = nib.Nifti1Image(lr_img, img.affine)
        name = os.path.basename(file_path)
        base_name = name.replace(".nii.gz", "_LR.nii.gz")
        save_path = os.path.join(output_dir, base_name)
        nib.save(lr_img_nib, save_path)
        saved_paths.append(save_path)
    return saved_paths

def extract_3D_patches(img, patch_size, stride):
    """
    Extracts 3D patches from a 3D image.
    
    Parameters:
    img (numpy array): The input 3D image.
    patch_size (tuple): The size of each patch (depth, height, width).
    stride (tuple): The stride for patch extraction (depth_stride, height_stride, width_stride).
    
    Returns:
    list: A list of 3D patches.
    """
    
    patches = []
    d, h, w = img.shape
    pd, ph, pw = patch_size
    sd, sh, sw = stride
    
    for z in range(0, d - pd + 1, sd):
        for y in range(0, h - ph + 1, sh):
            for x in range(0, w - pw + 1, sw):
                patch = img[z:z+pd, y:y+ph, x:x+pw] # can be clipped at edges
                patches.append(patch)
    
    return patches


def pad_to_shape(img, target_shape): 
    """
    Pads a 3D image to the target shape with zeros. Updates the affine matrix accordingly.
    
    Parameters:
    img (nibabel NIfTI image): The input 3D image.
    target_shape (tuple): The desired shape (depth, height, width).
    
    Returns:
    nibabel NIfTI image: The padded 3D image.
    """
    img_affine = img.affine
    img = img.get_fdata()

    pad_width = [( (t - s) // 2, (t - s) - (t - s) // 2 ) for s, t in zip(img.shape, target_shape)]
    padded_img = np.pad(img, pad_width, mode='constant', constant_values=0)
    
    new_affine = img_affine.copy()
    shift = np.array([pad[0] for pad in pad_width])
    new_affine[:3, 3] -= new_affine[:3, :3] @ shift

    padded_img = nib.Nifti1Image(padded_img, new_affine)

    return padded_img


def reconstruct_from_patches(patches, original_shape, stride):
    """
    Reconstructs the original 3D image from its patches.
    
    Parameters:
    patches (list): A list of 3D patches.
    original_shape (tuple): The shape of the original 3D image (depth, height, width).
    stride (tuple): The stride used during patch extraction (depth_stride, height_stride, width_stride).
    
    Returns:
    numpy array: The reconstructed 3D image.
    """
    
    reconstructed_img = np.zeros(original_shape)
    count_img = np.zeros(original_shape)  # To count overlaps
    
    pd, ph, pw = patches[0].shape
    sd, sh, sw = stride
    
    idx = 0
    for z in range(0, original_shape[0] - pd + 1, sd):
        for y in range(0, original_shape[1] - ph + 1, sh):
            for x in range(0, original_shape[2] - pw + 1, sw):
                reconstructed_img[z:z+pd, y:y+ph, x:x+pw] += patches[idx]
                count_img[z:z+pd, y:y+ph, x:x+pw] += 1
                idx += 1
    
    # Avoid division by zero
    count_img[count_img == 0] = 1
    reconstructed_img /= count_img
    
    return reconstructed_img


def get_patches(files, patch_size, stride, target_shape):
    """
    Extracts patches from the given files.

    Parameters:
    - files: List of tuples containing file paths for T1, T2, and T2_LR images.
    - patch_size: The size of each patch (depth, height, width).
    - stride: The stride for patch extraction (depth_stride, height_stride, width_stride).
    - target_shape: The target shape to which images will be padded.
    - ref_img: The reference image for resampling.

    Returns:
    list: List of T1 input patches.
    list: List of T2 output patches.
    list: List of T2_LR input patches.

    """
    t1_input = []
    t2_output = []
    t2_LR_input = []
    
    for t1_file, t2_file, t2_LR_file in files:
        #load images
        t1_img = nib.load(t1_file)
        t2_img = nib.load(t2_file)
        t2_LR_img = nib.load(t2_LR_file)
        #padding to be divisible by patch size
        t1_img = pad_to_shape(t1_img, target_shape)
        t2_img = pad_to_shape(t2_img, target_shape)
        t2_LR_img = pad_to_shape(t2_LR_img, target_shape)
        #normalizing
        t1_img = normalize(t1_img)
        t2_img = normalize(t2_img)
        t2_LR_img = normalize(t2_LR_img)
        #extracting patches
        t1_patches = extract_3D_patches(t1_img.get_fdata(), patch_size, stride)
        t2_patches = extract_3D_patches(t2_img.get_fdata(), patch_size, stride)
        t2_LR_patches = extract_3D_patches(t2_LR_img.get_fdata(), patch_size, stride)
        #add patches
        t1_input.append(t1_patches)
        t2_output.append(t2_patches)
        t2_LR_input.append(t2_LR_patches)
    
    return t1_input, t2_output, t2_LR_input

def normalize(img):

    """ Normalizes a NIfTI image to the range [0, 1] using the 1st and 99th percentiles.    
    Parameters:
    img (nibabel NIfTI image): The input NIfTI image to be normalized.

    Returns:
    nibabel NIfTI image: The normalized NIfTI image.
    """

    I = img.get_fdata()
    mask = I > 0  # Create a mask for non-zero values

    low_p, high_p = 1, 99
    min_val = np.percentile(I[mask], low_p)
    max_val = np.percentile(I[mask], high_p)

    eps = 1e-6  # Small epsilon to avoid division by zero
    normalized_img = (I - min_val) / (max_val - min_val + eps)
    normalized_img = np.clip(normalized_img, 0, 1)  # Clip values to [0, 1]

    return nib.Nifti1Image(normalized_img, affine=img.affine)

def save_images(file_list, output_dir):
    """
    Saves images as NIfTI files from the file list to the specified output directory.

    Parameters:
    - file_list: List of file paths.
    - output_dir: Directory where images will be saved.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for file_path in file_list:
        img = nib.load(file_path)
        img_data = img.get_fdata()
        base_name = os.path.basename(file_path)
        save_path = os.path.join(output_dir, base_name)
        nib.save(nib.Nifti1Image(img_data, img.affine), save_path)

def append_row(csv_path, row_dict):
    """
    Appends a row to a CSV file. If the file does not exist, it creates it and adds headers.
    
    Parameters:
    csv_path (str): Path to the CSV file.
    row_dict (dict): Dictionary representing the row to append, where keys are column names.
    """
    
    df = pd.DataFrame([row_dict])
    header = not os.path.exists(csv_path)
    df.to_csv(csv_path, mode="a", index=False, header=header)

def calculate_metrics(real_images, generated_images):
    """
    Calculate PSNR, SSIM, NRMSE, and MSE between real and generated images.
    
    Parameters:
    - real_images: List or array of ground truth images.
    - generated_images: List or array of generated images.
    
    Returns:
    - Dictionary with average PSNR, SSIM, NRMSE, and MSE values
    """
    psnr_list = []
    ssim_list = []
    nrmse_list = []
    mse_list = []
    for i in range(len(generated_images)): #adjust if different lengths
        p = peak_signal_noise_ratio(real_images[i], generated_images[i], data_range=1)
        psnr_list.append(p)
        s = structural_similarity(real_images[i], generated_images[i], data_range=1)
        ssim_list.append(s)
        n = normalized_root_mse(real_images[i], generated_images[i])
        nrmse_list.append(n)
        m = mean_squared_error(real_images[i], generated_images[i])
        mse_list.append(m)
    return {
        "psnr": np.mean(psnr_list),
        "ssim": np.mean(ssim_list),
        "nrmse": np.mean(nrmse_list),
        "mse": np.mean(mse_list)
    }