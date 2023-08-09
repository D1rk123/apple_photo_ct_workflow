#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 10:10:30 2023

@author: des
"""

from pathlib import Path
from tqdm import tqdm
import numpy as np

import ct_experiment_utils as ceu
from folder_locations import get_data_folder, get_results_folder

def crop_volume(img, mask):
    coords = np.stack(np.nonzero(mask))
    min_vec = np.clip(np.min(coords, axis=1)-10, 0, 10000)
    max_vec = np.max(coords, axis=1)+10
    
    return img[min_vec[0]:max_vec[0], min_vec[1]:max_vec[1], min_vec[2]:max_vec[2]]
    
if __name__ == "__main__":
    base_path = get_data_folder()
    recons_path = base_path / "fdk_bh_corrected_recons"
    masks_path = base_path / "masks2"
    out_path = base_path / "fdk_bh_corrected_recons_crop_tight"
    out_path.mkdir(exist_ok=True)
    
    for i in tqdm(range(1,121)):
        if not (recons_path / str(i)).exists() or \
           not (masks_path / str(i)).exists():
               print(f"Skipping {i}")
               continue
       
        recon = ceu.load_stack(recons_path / str(i))
        mask = ceu.load_stack(masks_path / str(i))
        crop = crop_volume(recon, mask)
        ceu.save_stack(out_path / str(i), crop)        
