#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 10:33:01 2022

@author: des
"""
import re

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

class PhotoMetadata():
    def __init__(self, included, excluded_slices, annotation_slice):
        self.included = included
        self.excluded_slices = excluded_slices
        self.annotation_slice = annotation_slice
        
    
def parse_metadata(photo_metadata_path):
    metadata_dict = {}
    with open(photo_metadata_path) as file:
        for i, line in enumerate(file):
            if i == 0:
                continue
            raw_str_split = line.rstrip().split(",")
            photo_nr = int(raw_str_split[0])
            included = int(raw_str_split[1]) != 0
            if raw_str_split[2] == "":
                excluded_slices = []
            else:
                excluded_slices = [int(s) for s in raw_str_split[2].split(" ")]
            if raw_str_split[3] == "":
                annotation_slice = None
            else:
                annotation_slice = int(raw_str_split[3])
            
            metadata_dict[photo_nr] = PhotoMetadata(included, excluded_slices, annotation_slice)
    return metadata_dict


def load_photo_data(photo_path, photo_mask_path, scan_name, excluded_slices):
    photo_nrs = []
    photo_masks = []
    photos = []
    
    photo_nrs_raw = sorted([int(re.findall(r'\d+', path.name)[1]) for path in photo_path.glob(f"Kanzi{scan_name}_slice_*.png")])

    for photo_nr in photo_nrs_raw:
        file_name = f"Kanzi{scan_name}_slice_{photo_nr}.png"
        photo_file = photo_path / file_name
        photo_mask_file = photo_mask_path / file_name
        if photo_nr not in excluded_slices and (photo_mask_file).exists():
            photo_nrs.append(photo_nr)
            mask_raw = skimage.io.imread(photo_mask_file)
            if mask_raw.ndim == 3:
                mask_raw = mask_raw[:, :, 0]
            photo_masks.append(np.flip(mask_raw < 128, axis=1))
            photos.append(np.flip(skimage.io.imread(photo_file), axis=1))
    
    return photo_nrs, photo_masks, photos
    
    
def get_photo_coords(photo_mask, crop_info, edge_discard_distance, bbox_data):
    content_min_extents, content_max_extents = crop_info.get_crop_content_extents()
    
    clipping_mask = np.zeros((bbox_data.crop_size[0], bbox_data.crop_size[1]), dtype=np.uint8)
    clipping_mask[content_min_extents[0]+edge_discard_distance:content_max_extents[0]-edge_discard_distance,
                  content_min_extents[1]+edge_discard_distance:content_max_extents[1]-edge_discard_distance] = 1
                  
    photo_values = photo_mask[
        content_min_extents[0]+edge_discard_distance:content_max_extents[0]-edge_discard_distance,
        content_min_extents[1]+edge_discard_distance:content_max_extents[1]-edge_discard_distance]
    photo_values = torch.from_numpy(photo_values.flatten())

    photo_clipping_mask_coords = torch.argwhere(torch.from_numpy(clipping_mask))
    photo_clipping_mask_coords = slice_to_volume_coords(photo_clipping_mask_coords.to(dtype=torch.float32))
    
    photo_edge_mask = (photo_mask ^ skimage.morphology.binary_erosion(photo_mask))*clipping_mask
    photo_edge_coords = torch.argwhere(torch.from_numpy(photo_edge_mask)).to(dtype=torch.float32)
    photo_edge_coords = slice_to_volume_coords(photo_edge_coords)

    photo_full_coords = torch.argwhere(torch.ones(photo_mask.shape)).to(dtype=torch.float32)
    photo_full_coords = slice_to_volume_coords(photo_full_coords)

    return photo_clipping_mask_coords, photo_edge_coords, photo_full_coords, photo_values

def get_photo_coords_lists(scan_number, photo_nrs, photo_masks, edge_discard_distance, bbox_data):
    photo_full_coords_list = []
    photo_edge_coords_list = []
    photo_clipping_mask_coords_list = []
    photo_mask_values_list = []
    
    for photo_nr, photo_mask in zip(photo_nrs, photo_masks):
        crop_info = bbox_data.get_crop_info(scan_number, photo_nr)
        photo_clipping_mask_coords, photo_edge_coords, photo_full_coords, photo_values = get_photo_coords(photo_mask, crop_info, edge_discard_distance, bbox_data)
        
        photo_clipping_mask_coords_list.append(photo_clipping_mask_coords)
        photo_edge_coords_list.append(photo_edge_coords)
        photo_full_coords_list.append(photo_full_coords)
        photo_mask_values_list.append(photo_values)
        
    return photo_full_coords_list, photo_edge_coords_list, photo_clipping_mask_coords_list, photo_mask_values_list
    
def get_learning_rates(multiplier, num_photos):
    angles_lr = (torch.tensor([1e-5, 1e-5, 1e-5], dtype=torch.float32)/5)*multiplier
    slice_thickness_lr = (1e-3)*multiplier
    slice_offset_lr = (4e-2)*multiplier
    scale_lr = (2e-8)*multiplier
    slice_xy_offset_lr = 1*multiplier
    
    if num_photos == 1:
        return [angles_lr, slice_offset_lr, scale_lr] + ([slice_xy_offset_lr] * num_photos)
    else:
        return [angles_lr, slice_thickness_lr, slice_offset_lr, scale_lr] + ([slice_xy_offset_lr] * num_photos)

def plot_registered_profiles(write_path, ct_profile, photo_profile, photo_nrs, tm):
    plt.figure()
    plt.plot(range(len(ct_profile.numpy())), ct_profile.numpy())
    plt.plot((-torch.tensor(photo_nrs)*tm.slice_thickness + tm.slice_offset).cpu().numpy(), (photo_profile*tm.scale).cpu().numpy())
    plt.title(f"Initialization, MSE = {mse}")
    plt.savefig(write_path)
    plt.close()
    
def plot_registration_result(write_path, photo_nr, scan_name, photo, registered_ct_slice):
    combined_img = np.zeros(photo.shape, dtype=np.float32)
    combined_img[:, :, 1] = np.mean(photo.astype(np.float32), axis=2)
    combined_img[:, :, 1] /= np.max(combined_img[:, :, 1])
    combined_img[:, :, 0] = (registered_ct_slice/torch.max(registered_ct_slice)).numpy()
    combined_img[:, :, 2] = combined_img[:, :, 0]
    np.clip(combined_img, 0, 1, out=combined_img)
    
    plt.figure(figsize=(13, 5))
    plt.subplot(131)
    plt.imshow(photo)
    plt.title(f"Photograph {photo_nr} of apple {scan_name}")
    plt.subplot(132)
    plt.imshow(registered_ct_slice.numpy(), cmap="gist_gray")
    plt.title("Registered CT slice")
    plt.subplot(133)
    plt.imshow(combined_img)
    plt.title("Combined (Photo=green, CT=purple)")
    plt.tight_layout()
    plt.savefig(write_path)
    plt.close()

if __name__ == "__main__":
    #Stop plots from displaying
    #matplotlib.use('Agg')

    scans_folder = get_data_folder()
    photo_metadata_path = scans_folder / "photo_metadata.csv"
    bbox_data_path = scans_folder / "slice_photos" / "photo_bboxes.csv"
    # output folders
    experiment_folder = ceu.make_new_experiment_folder(get_results_folder(), name="register_kanzi_dataset_MSE_overlap")
    registration_parameters_folder = experiment_folder / "registration_parameters"
    registration_parameters_folder.mkdir(exist_ok=True)
    initialization_parameters_folder = experiment_folder / "initialization_parameters"
    initialization_parameters_folder.mkdir(exist_ok=True)
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
        registration_parameters_path = registration_parameters_folder / f"registration_params_{scan_name}.txt"
        initialization_parameters_path = initialization_parameters_folder / f"registration_params_{scan_name}.txt"
        error_log_path = error_log_folder / f"error_log_{scan_name}.txt"
        metadata = metadata_dict[scan_number]
        
        if not metadata.included:
            continue

        photo_nrs, photo_masks, photos = load_photo_data(photo_path, photo_mask_path, scan_name, metadata.excluded_slices)
        
        ct_mask = torch.from_numpy(ceu.load_stack(ct_mask_path))
        ct_recon = torch.from_numpy(ceu.load_stack(recon_path))
        
        registration = PhotoCtRegistration(ParallelSliceSimilarityTM(photo_nrs))

        mse, ct_profile, photo_profile = registration.initialize_on_profiles(
            ct_mask,
            photo_masks,
            num_iterations=10000
        )
        plot_registered_profiles(
            registration_figures_path / "initialization.png",
            ct_profile,
            photo_profile,
            photo_nrs,
            registration.tm)
        
        registration.initialize_xy_offsets(ct_mask, photo_masks)
        registration.tm.write(initialization_parameters_path)
        
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
        cost_function = OverlapMseCost(
            photo_nrs,
            photo_clipping_mask_coords_list,
            photo_mask_values_list,
            ct_mask,
        ) 
        registration.optimize_cost_function(
            cost_function = cost_function,
            learning_rates = get_learning_rates(2000, len(photo_nrs)),
            momentum = 0.75,
            max_iterations = 100000,
            device = device,
            error_log_path = error_log_path,
            stopping_condition = stopping_condition,
            verbose = False
            )
        """
        # Alternatively you can use an edge based cost function   
        cost_function = EdgeL1DistanceCost(
            photo_nrs,
            photo_edge_coords_list,
            ct_dm,
            verbose=False
        )
        registration.optimize_cost_function(
            cost_function = cost_function,
            learning_rates = get_learning_rates(3, len(photo_nrs)),
            momentum = 0.75,
            max_iterations = 200000,
            device = device,
            error_log_path = error_log_path,
            stopping_condition = stopping_condition
            )
        """
        registration.tm.write(registration_parameters_path)
        
        for photo, photo_nr, photo_full_coords in zip(photos, photo_nrs, photo_full_coords_list):
            transformed_coords = registration.tm.transform([photo_nr], [photo_full_coords])[0]
            ct_lookup = volume_lookup(ct_recon, transformed_coords.cpu())
            registered_ct_slice = ct_lookup.reshape(photo.shape[0:2])
                        
            plot_registration_result(
                registration_figures_path / f"{photo_nr}.png",
                photo_nr,
                scan_name,
                photo,
                registered_ct_slice)
