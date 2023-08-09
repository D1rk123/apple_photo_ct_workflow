#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 11:25:08 2023

@author: des
"""
import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

import ct_experiment_utils as ceu
from folder_locations import get_data_folder, get_results_folder
from process_apple_bboxes import BboxDataParser
from photo_ct_registration import PhotoCtRegistration
from cost_functions import OverlapMseCost
from image_coordinate_utils import volume_lookup, slice_to_volume_coords
from register_kanzi_dataset import parse_metadata, load_photo_data, get_photo_coords_lists, get_learning_rates, plot_registration_result
from transformation_models import ParallelSliceSimilarityTM

if __name__ == "__main__":
    #Stop plots from displaying
    #matplotlib.use('Agg')

    scans_folder = get_data_folder()
    photo_metadata_path = scans_folder / "photo_metadata.csv"
    bbox_data_path = scans_folder / "slice_photos" / "photo_bboxes.csv"
    # output folders
    experiment_folder = ceu.make_new_experiment_folder(get_results_folder(), name="register_kanzi_dataset_subsets")
    registration_parameters_folder = experiment_folder / "registration_parameters"
    registration_parameters_folder.mkdir()
    registration_figures_folder = experiment_folder / "registration_figures"
    registration_figures_folder.mkdir()
    error_log_folder = experiment_folder / "error_logs"
    error_log_folder.mkdir()

    bbox_data = BboxDataParser(bbox_data_path, full_size=(1942, 2590), edge_clamp_range=50)
    metadata_dict = parse_metadata(photo_metadata_path)
    
    edge_discard_distance = 10
    device = "cuda"

    for scan_number in range(9, 121):
        print(f"Registering scan {scan_number}")
        scan_name = str(scan_number)
        recon_path = scans_folder / "fdk_bh_corrected_recons" / scan_name
        ct_mask_path = scans_folder / "masks2" / scan_name
        photo_path = scans_folder / "slice_photos_crop" / scan_name
        photo_mask_path = scans_folder / "slice_photo_masks_crop_all" / scan_name

        metadata = metadata_dict[scan_number]
        
        if not metadata.included:
            continue

        photo_nrs, photo_masks, photos = load_photo_data(photo_path, photo_mask_path, scan_name, metadata.excluded_slices)
        
        near_annotation_nrs = [metadata.annotation_slice+i for i in [0, -1, 1, -2, 2, -3]]
        if any([i not in photo_nrs for i in near_annotation_nrs]):
            continue
        ct_mask = torch.from_numpy(ceu.load_stack(ct_mask_path))
        ct_recon = torch.from_numpy(ceu.load_stack(recon_path))
        
        full_registration = PhotoCtRegistration(ParallelSliceSimilarityTM(photo_nrs))

        mse, ct_profile, photo_profile = full_registration.initialize_on_profiles(
            ct_mask,
            photo_masks,
            num_iterations=10000
        )
        
        full_registration.initialize_xy_offsets(ct_mask, photo_masks)
        
        photo_full_coords_list, _, \
            photo_clipping_mask_coords_list, photo_mask_values_list \
            = get_photo_coords_lists(
                scan_number, photo_nrs, photo_masks,
                edge_discard_distance, bbox_data)
            
        for num_slices in range(1, len(near_annotation_nrs)+1):
            photo_nrs_subset = sorted(near_annotation_nrs[:num_slices])
            subset_indices = [photo_nrs.index(i) for i in photo_nrs_subset]
            photos_subset = [photos[i] for i in subset_indices]
            photo_full_coords_subset = [photo_full_coords_list[i] for i in subset_indices]
            photo_clipping_mask_coords_subset = [photo_clipping_mask_coords_list[i] for i in subset_indices]
            photo_mask_values_subset = [photo_mask_values_list[i] for i in subset_indices]
            
            registration = PhotoCtRegistration(full_registration.tm.take_subset(photo_nrs_subset))
            
            registration_figures_path =  registration_figures_folder / scan_name / f"{num_slices}slices"
            registration_figures_path.mkdir(parents=True)
            registration_parameters_path = registration_parameters_folder / f"{num_slices}slices" / f"registration_params_{scan_name}.txt"
            registration_parameters_path.parent.mkdir(exist_ok=True)
            error_log_path = error_log_folder / f"{num_slices}slices" / f"error_log_{scan_name}.txt"
            error_log_path.parent.mkdir(exist_ok=True)
            
            def stopping_condition(cost_log):
                if len(cost_log)<1001:
                    return False
                if max(cost_log[-1000:])-min(cost_log[-1000:])<0.00001:
                    return True
                return False
            cost_function = OverlapMseCost(
                photo_nrs_subset,
                photo_clipping_mask_coords_subset,
                photo_mask_values_subset,
                ct_mask,
            ) 
            registration.optimize_cost_function(
                cost_function = cost_function,
                learning_rates = get_learning_rates(500, num_slices),
                momentum = 0.75,
                max_iterations = 100000,
                device = device,
                error_log_path = error_log_path,
                stopping_condition = stopping_condition,
                verbose = False
                )
            
            registration.tm.write(registration_parameters_path)
            for photo, photo_nr, photo_full_coords in zip(photos_subset, photo_nrs_subset, photo_full_coords_subset):
                transformed_coords = registration.tm.transform([photo_nr], [photo_full_coords])[0]
                ct_lookup = volume_lookup(ct_recon, transformed_coords.cpu())
                registered_ct_slice = ct_lookup.reshape(photo.shape[0:2])
                            
                plot_registration_result(
                    registration_figures_path / f"{photo_nr}.png",
                    photo_nr,
                    scan_name,
                    photo,
                    registered_ct_slice)
        
