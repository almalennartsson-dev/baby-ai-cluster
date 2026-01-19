import numpy as np
import torch
import torch.nn as nn
from skimage.metrics import structural_similarity, peak_signal_noise_ratio, mean_squared_error, normalized_root_mse

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

