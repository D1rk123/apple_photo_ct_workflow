#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 10:33:01 2022

@author: des
"""
import re
import sys
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt
import skimage.morphology
import skimage.transform
import skimage.io

import ct_experiment_utils as ceu
from folder_locations import get_data_folder, get_results_folder
from process_apple_bboxes import BboxDataParser
from photo_ct_registration import PhotoCtRegistration
from cost_functions import EdgeL1DistanceCost, OverlapMseCost
from image_coordinate_utils import volume_lookup, slice_to_volume_coords
from transformation_models import ParallelSliceSimilarityTM
from register_kanzi_dataset import parse_metadata, load_photo_data, get_photo_coords_lists, get_learning_rates, plot_registration_result

if __name__ == "__main__":
    #Stop plots from displaying
    #matplotlib.use('Agg')

    scans_folder = get_data_folder()
    photo_metadata_path = scans_folder / "photo_metadata.csv"
    bbox_data_path = scans_folder / "slice_photos" / "photo_bboxes.csv"
    # output folders
    results_folder = get_results_folder()
    experiment_folder = ceu.make_new_experiment_folder(get_results_folder(), name="register_kanzi_dataset_separate")
    registration_parameters_folder = experiment_folder / "registration_parameters"
    registration_parameters_folder.mkdir(exist_ok=True)
    initialization_parameters_folder = Path(sys.argv[1])
    registration_figures_folder = experiment_folder / "registration_figures"
    registration_figures_folder.mkdir(exist_ok=True)
    error_log_folder = experiment_folder / "error_logs"
    error_log_folder.mkdir(exist_ok=True)

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
        registration_figures_path =  registration_figures_folder / scan_name
        registration_figures_path.mkdir(exist_ok=True)
        registration_parameters_subfolder = registration_parameters_folder / scan_name
        registration_parameters_subfolder.mkdir()
        initialization_parameters_path = initialization_parameters_folder / f"registration_params_{scan_name}.txt"
        error_log_subfolder = error_log_folder / scan_name
        error_log_subfolder.mkdir()
        metadata = metadata_dict[scan_number]
        
        if not metadata.included:
            continue

        photo_nrs, photo_masks, photos = load_photo_data(photo_path, photo_mask_path, scan_name, metadata.excluded_slices)
        
        ct_mask = torch.from_numpy(ceu.load_stack(ct_mask_path))
        ct_recon = torch.from_numpy(ceu.load_stack(recon_path))
        
        tm_start = ParallelSliceSimilarityTM(path=initialization_parameters_path)
        
        photo_full_coords_list, photo_edge_coords_list, \
            photo_clipping_mask_coords_list, photo_mask_values_list \
            = get_photo_coords_lists(
                scan_number, photo_nrs, photo_masks,
                edge_discard_distance, bbox_data)
        
        
        def stopping_condition(cost_log):
            if len(cost_log)<1001:
                return False
            if max(cost_log[-1000:])-min(cost_log[-1000:])<0.00001:
                return True
            return False
        
        for photo, photo_nr, photo_full_coords, photo_clipping_mask_coords, photo_mask_values in zip(photos, photo_nrs, photo_full_coords_list, photo_clipping_mask_coords_list, photo_mask_values_list):
            registration_parameters_path = registration_parameters_subfolder / f"registration_params_{scan_name}_{photo_nr}.txt"
            error_log_path = error_log_subfolder / f"error_log_{scan_name}_{photo_nr}.txt"
            
            registration = PhotoCtRegistration(tm_start.take_subset([photo_nr]))
        
            cost_function = OverlapMseCost(
                [photo_nr],
                [photo_clipping_mask_coords],
                [photo_mask_values],
                ct_mask,
            ) 
            registration.optimize_cost_function(
                cost_function = cost_function,
                learning_rates = get_learning_rates(500, 1),
                momentum = 0.75,
                max_iterations = 100000,
                device = device,
                error_log_path = error_log_path,
                stopping_condition = stopping_condition,
                verbose = False
                )
            registration.tm.write(registration_parameters_path)
                
            transformed_coords = registration.tm.transform([photo_nr], [photo_full_coords])[0]
            ct_lookup = volume_lookup(ct_recon, transformed_coords.cpu())
            registered_ct_slice = ct_lookup.reshape(photo.shape[0:2])
                        
            plot_registration_result(
                registration_figures_path / f"{photo_nr}.png",
                photo_nr,
                scan_name,
                photo,
                registered_ct_slice)
