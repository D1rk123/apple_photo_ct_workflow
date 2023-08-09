#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 17:01:19 2023

@author: des
"""
import re

import numpy as np
import tifffile
import matplotlib.pyplot as plt
import skimage.measure
import skimage.io
import scipy

import ct_experiment_utils as ceu
from folder_locations import get_data_folder, get_results_folder

def postprocess_mask(mask, fill_holes):
    mask = mask > 0.5
    #mask = skimage.morphology.binary_erosion(mask, skimage.morphology.disk(10))
    # find the largest connected component
    # https://stackoverflow.com/questions/47540926/get-the-largest-connected-component-of-segmentation-image
    labels = skimage.measure.label(mask)
    if labels.max() == 0: #if there is no CC just return the  (all zero) mask
        return mask
        
    mask = (labels == np.argmax(np.bincount(labels.flat)[1:])+1)
    #mask = skimage.morphology.binary_dilation(mask, skimage.morphology.disk(10))
    if fill_holes:
        mask = scipy.ndimage.binary_fill_holes(mask)

    return mask

def read_photo_metadata(path):
    excluded_indices = []
    with open(path) as file:
        for i, line in enumerate(file):
            if i == 0:
                continue
            picture_numbers = (line.split(",")[2]).split(" ")
            if picture_numbers[0] == "":
                continue
            for picture_number in picture_numbers:
                excluded_indices.append((i, int(picture_number)))
    return excluded_indices

def find_mask_indices(masks_path):
    picture_indices = []

    for i in range(1, 121):
        files = list((masks_path / f"{i}").glob("Kanzi*_slice_*.png"))
        for file in files:
            slice_number = int(re.findall('\d+', file.name)[-1])
            picture_indices.append((i, slice_number))

    return picture_indices
    
def save_photo_overlay(photo, mask, save_path, scan_nr, photo_nr):
    photo[:, :, 2] = np.maximum(mask*255, photo[:, :, 2])
    
    plt.figure(figsize=(13, 5))
    plt.subplot(121)
    plt.imshow(mask)
    plt.title(f"Automatic segmentation of apple {scan_nr} photo {photo_nr}")
    #plt.subplot(132)
    #plt.imshow(mask_diff)
    #plt.title("mask difference")
    #plt.colorbar()
    plt.subplot(122)
    plt.imshow(photo)
    plt.title("Photo with segmentation overlay")
    plt.savefig(save_path)
    plt.close()
    #plt.show()

if __name__ == '__main__':
    scans_folder = get_data_folder()
    photos_path = scans_folder / "slice_photos_crop"
    masks_path = scans_folder / "slice_photo_masks_crop"
    save_folder = get_results_folder()
    experiment_folder = ceu.make_new_experiment_folder(save_folder, name="nn_output_combination")
    nn_output_folders = [
        save_folder / "split0_2023-01-23_picture_slice_segmentation_1" / "slice_photo_masks_crop_nn_output",
        save_folder / "split1_2023-01-23_picture_slice_segmentation_1" / "slice_photo_masks_crop_nn_output",
        save_folder / "split2_2023-01-23_picture_slice_segmentation_1" / "slice_photo_masks_crop_nn_output",
        save_folder / "split3_2023-01-23_picture_slice_segmentation_1" / "slice_photo_masks_crop_nn_output",
        save_folder / "split4_2023-01-30_picture_slice_segmentation_1" / "slice_photo_masks_crop_nn_output",
        ]
    
    split_apple_nrs = [
        [40, 34, 43, 9, 21, 91, 11, 24, 31, 47, 64, 74, 83, 99, 107],
        [58, 49, 50, 15, 25, 92, 12, 26, 32, 51, 66, 76, 85, 100, 113],
        [39, 69, 19, 82, 28, 96, 13, 27, 33, 54, 67, 77, 87, 101, 115],
        [88, 56, 97, 110, 37, 102, 17, 29, 42, 57, 70, 78, 90, 103, 116],
        [62, 68, 71, 63, 20, 81, 10, 22, 30, 44, 60, 73, 80, 95, 106, 119]
    ]
    all_split_apple_nrs = [nr for split in split_apple_nrs for nr in split]
    test_apple_nrs = [nr for nr in range(9, 121) if nr not in all_split_apple_nrs]
    included_scan_nrs = list(range(9, 121))

    excluded_indices = read_photo_metadata(scans_folder / "photo_metadata.csv")
    mask_indices = find_mask_indices(masks_path)
    excluded_indices += mask_indices
    
    picture_indices = []
    for scan_nr in included_scan_nrs:
        files = list((photos_path / f"{scan_nr}").glob("Kanzi*_slice_*.png"))
        photo_nrs = sorted([int(re.findall('\d+', file.name)[-1]) for file in files])
        for photo_nr in photo_nrs:
            index = (scan_nr, photo_nr)
            if not index in excluded_indices:
                picture_indices.append(index)
    
    num_pixels_total = 1966*1856
    for scan_nr, photo_nr in picture_indices:
        print((scan_nr, photo_nr))
        nn_outputs = np.stack([
            tifffile.imread(str(folder / f"{scan_nr}" / f"Kanzi{scan_nr}_slice_{photo_nr}.tif"))
            for folder in nn_output_folders])
        #thresh_first_results = np.sum(nn_outputs>0.5, axis=0)
        #thresh_first_result = thresh_first_results >= 3
        thresh_after_result = postprocess_mask(np.average(nn_outputs, axis=0), photo_nr>3)
        #print(f"{scan_nr},{photo_nr}," + ",".join([f"frac({i})={np.sum(thresh_first_results==i)/num_pixels_total:.04f}" for i in range(0,6)]))
        
        #photo = skimage.io.imread(photos_path / f"{scan_nr}" / f"Kanzi{scan_nr}_slice_{photo_nr}.png")
        #plot_folder = experiment_folder / "registration_plots" / f"{scan_nr}"
        #plot_folder.mkdir(exist_ok=True, parents=True)
        #save_photo_overlay(photo, thresh_after_result, plot_folder/f"Kanzi{scan_nr}_slice_{photo_nr}.png", scan_nr, photo_nr)
        nn_mask_folder = experiment_folder / "masks_crop_nn" / f"{scan_nr}"
        nn_mask_folder.mkdir(exist_ok=True, parents=True)
        result_uint8 = (1-thresh_after_result.astype(np.uint8))*255
        skimage.io.imsave(nn_mask_folder / f"Kanzi{scan_nr}_slice_{photo_nr}.png", result_uint8)
        
        
        
                                              
        
    
    
