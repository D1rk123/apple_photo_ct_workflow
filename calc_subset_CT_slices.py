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
    # output folders
    results_folder = get_results_folder()
    experiment_folder = ceu.make_new_experiment_folder(get_results_folder(), name="subset_CT_slices")
    output_folders = [experiment_folder / f"{i}slices" for i in range(1, 7)]
    for folder in output_folders:
        folder.mkdir()
    params_output_folders = [f / "registration_parameters"  for f in output_folders]
    for folder in params_output_folders:
        folder.mkdir()
    
    subset1_3_path = results_folder / "2023-04-04_register_kanzi_dataset_subsets_1to3_1" / "registration_parameters"
    subset4_6_path = results_folder / "2023-04-04_register_kanzi_dataset_subsets_4to6_1" / "registration_parameters"
    registration_params_folders = (
        [subset1_3_path / f"{i}slices" for i in range(1, 4)]
        + [subset4_6_path / f"{i}slices" for i in range(4, 7)]
    )

    bbox_data = BboxDataParser(bbox_data_path, full_size=(1942, 2590), edge_clamp_range=50)
    metadata_dict = parse_metadata(photo_metadata_path)
    
    edge_discard_distance = 10
    device = "cuda"

    for scan_number in range(9, 121):
        scan_name = str(scan_number)
        recon_path = scans_folder / "fdk_bh_corrected_recons" / scan_name
        ct_mask_path = scans_folder / "masks2" / scan_name
        photo_path = scans_folder / "slice_photos_crop" / scan_name
        photo_mask_path = scans_folder / "slice_photo_masks_crop_all" / scan_name
        metadata = metadata_dict[scan_number]
        
        if not metadata.included:
            continue
            
        if not (registration_params_folders[-1] / f"registration_params_{scan_name}.txt").exists():
            continue
        
        photo_nrs, photo_masks, photos = load_photo_data(photo_path, photo_mask_path, scan_name, metadata.excluded_slices)
        
        ct_recon = torch.from_numpy(ceu.load_stack(recon_path))
        
        photo_full_coords_list, _, \
            _, _ \
            = get_photo_coords_lists(
                scan_number, photo_nrs, photo_masks,
                edge_discard_distance, bbox_data)
        
        for registration_params_folder, output_folder, params_output_folder in zip(registration_params_folders, output_folders, params_output_folders):
            photo_nr = metadata.annotation_slice
            photo_nr_i = photo_nrs.index(photo_nr)
                    
            tm = ParallelSliceSimilarityTM(path=registration_params_folder / f"registration_params_{scan_name}.txt")
            tm.write(params_output_folder / f"registration_params_{scan_name}.txt")
                
            transformed_coords = tm.transform([photo_nr], photo_full_coords_list[photo_nr_i:photo_nr_i+1])[0]
            ct_lookup = volume_lookup(ct_recon, transformed_coords.cpu())
            registered_ct_slice = ct_lookup.reshape(photos[0].shape[0:2])
                        
            tifffile.imwrite(str(output_folder / f"Kanzi{scan_name}_slice_{photo_nr}.tiff"), registered_ct_slice.numpy())
