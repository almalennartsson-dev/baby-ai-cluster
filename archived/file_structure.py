
import os, csv
from pathlib import Path
import random
import numpy as np
from scipy.ndimage import zoom
import nibabel as nib
import pandas as pd

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