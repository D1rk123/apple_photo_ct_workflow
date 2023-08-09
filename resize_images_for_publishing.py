#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 15:47:46 2023

@author: des

Downscales images and saves them in appropriate file formats for a good trade-off between image quality and document size
"""
from pathlib import Path
from tqdm import tqdm

import numpy as np
import skimage.morphology
import skimage.transform
import skimage.io
import tifffile
from matplotlib import pyplot as plt

import ct_experiment_utils as ceu
from folder_locations import get_data_folder, get_results_folder

def float_to_grayscale(img, lower, upper):
    img = np.clip(img, lower, upper)
    mapped = (((img-lower)/(upper-lower))*255).astype(np.uint8)
    return mapped


def make_checkerboard(img1, img2, width, height):
    checker_mask = np.mgrid[0:img1.shape[0], 0:img1.shape[1]]
    checker_mask = np.logical_xor(checker_mask[0]//width%2, checker_mask[1]//height%2)
    return img1*checker_mask[:,:,None]+img2*np.logical_not(checker_mask[:,:,None])

    
def make_purple_green(ct, photo):
    result = np.zeros_like(photo)
    result[:,:,(0,2)] = ct[:,:,None]
    result[:,:,1] = np.mean(photo, axis=2)
    return result

if __name__ == "__main__":
    photos_folder = get_data_folder() / "slice_photos_crop"
    photo_masks_folder = get_data_folder() / "slice_photo_masks_crop_all"
    ct_slices_folder = get_results_folder() / "2023-07-10_calc_kanzi_ct_slices_1" / "registered_ct_slices"
    
    experiment_folder = ceu.make_new_experiment_folder(get_results_folder(), name="postprocess_images_for_publishing")
    photos_out_folder = experiment_folder / "photos"
    photo_masks_out_folder = experiment_folder / "photo_masks"
    ct_out_folder = experiment_folder / "ct_slices"
    combined_out_folder = experiment_folder / "combined"
    
    for apple in tqdm(range(9, 121)):
        slice_indices = [i for i in range(25) if \
                         (ct_slices_folder / f"{apple}" / f"Kanzi{apple}_slice_{i}.tiff").exists() and \
                         (photos_folder / f"{apple}" / f"Kanzi{apple}_slice_{i}.png").exists()]
        (photos_out_folder / f"{apple}").mkdir(parents=True)
        (ct_out_folder / f"{apple}").mkdir(parents=True)
        (combined_out_folder / f"{apple}").mkdir(parents=True)
        (photo_masks_out_folder / f"{apple}").mkdir(parents=True)
        
        for i in slice_indices:
            ct_file_path = ct_slices_folder / f"{apple}" / f"Kanzi{apple}_slice_{i}.tiff"
            if ct_file_path.exists():
                ct_slice = tifffile.imread(ct_file_path)
                ct_slice = np.flip(ct_slice, axis=1)
                ct_slice = skimage.transform.resize(ct_slice, (566, 600), anti_aliasing=True)
                ct_slice = float_to_grayscale(ct_slice, 0, 0.95)
                skimage.io.imsave(ct_out_folder / f"{apple}" / f"Kanzi{apple}_slice_{i}.png", ct_slice)
                
            photo_file_path = photos_folder / f"{apple}" / f"Kanzi{apple}_slice_{i}.png"
            if photo_file_path.exists():
                photo = skimage.io.imread(photo_file_path)
                photo = np.round(skimage.transform.resize(photo, (566, 600, 3), anti_aliasing=True)*255).astype(np.uint8)
                skimage.io.imsave(photos_out_folder / f"{apple}" / f"Kanzi{apple}_slice_{i}.jpg", photo, quality=95)
            
            photo_mask_file_path = photo_masks_folder / f"{apple}" / f"Kanzi{apple}_slice_{i}.png"
            if photo_mask_file_path.exists():
                photo_mask = skimage.io.imread(photo_mask_file_path)
                if(len(photo_mask.shape)==3):
                    photo_mask = photo_mask[:,:,0]
                photo_mask = (photo_mask < 0.5).astype(float)
                photo_mask = np.round(skimage.transform.resize(photo_mask, (566, 600), anti_aliasing=True)*255).astype(np.uint8)
                skimage.io.imsave(photo_masks_out_folder / f"{apple}" / f"Kanzi{apple}_slice_{i}.png", photo_mask)
            
            if ct_file_path.exists() and photo_file_path.exists():
                combined = make_purple_green(ct_slice, photo)
                skimage.io.imsave(combined_out_folder / f"{apple}" / f"Kanzi{apple}_slice_{i}.png", combined)
            
            
        
    
    
