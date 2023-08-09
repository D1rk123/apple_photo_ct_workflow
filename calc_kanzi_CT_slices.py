#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 10:33:01 2022

@author: des
"""
from pathlib import Path
import numpy as np
import torch
import tifffile

import ct_experiment_utils as ceu
from folder_locations import get_data_folder, get_results_folder
from process_apple_bboxes import BboxDataParser
from image_coordinate_utils import volume_lookup
from transformation_models import ParallelSliceSimilarityTM
from register_kanzi_dataset import parse_metadata, load_photo_data, get_photo_coords_lists

if __name__ == "__main__":
    #Stop plots from displaying
    #matplotlib.use('Agg')

    scans_folder = get_data_folder()
    photo_metadata_path = scans_folder / "photo_metadata.csv"
    bbox_data_path = scans_folder / "slice_photos" / "photo_bboxes.csv"
    registration_parameters_folder_in = get_results_folder() / "2023-04-07_register_kanzi_dataset_MSE_overlap_1" / "registration_parameters"
    # output folders
    experiment_folder = ceu.make_new_experiment_folder(get_results_folder(), name="calc_kanzi_ct_slices")
    registration_parameters_folder = experiment_folder / "registration_parameters"
    registration_parameters_folder.mkdir(exist_ok=True)

    with open(photo_metadata_path) as file:
        photo_metadata = [line.rstrip() for line in file]

    bbox_data = BboxDataParser(bbox_data_path, full_size=(1942, 2590), edge_clamp_range=50)
    metadata_dict = parse_metadata(photo_metadata_path)
    
    edge_discard_distance = 10
    only_annotation_slice = False

    for scan_number in range(9, 121):
        metadata = metadata_dict[scan_number]
        
        if not metadata.included:
            continue
    
        photo_shape = (1856,1966)
        scan_name = str(scan_number)
        recon_path = scans_folder / "fdk_bh_corrected_recons" / scan_name
        ct_mask_path = scans_folder / "masks2" / scan_name
        photo_path = scans_folder / "slice_photos_crop" / scan_name
        photo_mask_path = scans_folder / "slice_photo_masks_crop_all" / scan_name
        registration_parameters_path = registration_parameters_folder / f"registration_params_{scan_name}.txt"
        registered_ct_slices_folder = experiment_folder / "registered_ct_slices" / scan_name
        registered_ct_slices_folder.mkdir(parents=True)
        
        photo_nrs, photo_masks, photos = load_photo_data(photo_path, photo_mask_path, scan_name, metadata.excluded_slices)
        
        ct_recon = torch.from_numpy(ceu.load_stack(recon_path))
        
        photo_full_coords_list, _, \
            _, _ \
            = get_photo_coords_lists(
                scan_number, photo_nrs, photo_masks,
                edge_discard_distance, bbox_data)
                
        tm = ParallelSliceSimilarityTM(path=registration_parameters_folder_in / f"registration_params_{scan_name}.txt")
        tm.write(registration_parameters_path)
        
        transformed_coords_list = tm.transform(photo_nrs, photo_full_coords_list)
        
        for photo_nr, transformed_coords in zip(photo_nrs, transformed_coords_list):
            if only_annotation_slice and photo_nr != metadata.annotation_slice:
                continue
            
            ct_lookup = volume_lookup(ct_recon, transformed_coords)
            registered_ct_slice = ct_lookup.reshape(photos[0].shape[0:2])
                        
            tifffile.imwrite(str(registered_ct_slices_folder / f"Kanzi{scan_name}_slice_{photo_nr}.tiff"), registered_ct_slice.numpy())
            
        
