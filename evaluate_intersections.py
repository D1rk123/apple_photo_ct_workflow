#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 15:32:05 2023

@author: des
"""
import numpy as np
import numpy.random
import scipy
from scipy.spatial import Delaunay
import skimage.morphology
import torch
from tqdm import tqdm

import ct_experiment_utils as ceu
from folder_locations import get_data_folder, get_results_folder
from process_apple_bboxes import BboxDataParser
from transformation_models import ParallelSliceSimilarityTM
from image_coordinate_utils import slice_to_volume_coords
from register_kanzi_dataset import parse_metadata, load_photo_data

def get_photo_coords(photo_mask):
    edge_mask = (photo_mask ^ skimage.morphology.binary_dilation(photo_mask))
    # zero every other column and row to avoid precision problems in the convex hull test
    # it shouldn't affect the results
    edge_mask[::2,:] = 0
    edge_mask[:,::2] = 0
    edge_coords = torch.argwhere(torch.from_numpy(edge_mask)).to(dtype=torch.float32)
    edge_coords = slice_to_volume_coords(edge_coords)

    mask_coords = torch.argwhere(torch.from_numpy(photo_mask.copy())).to(dtype=torch.float32)
    mask_coords = slice_to_volume_coords(mask_coords)

    return edge_coords, mask_coords
    
def get_photo_coords_lists(scan_number, photo_nrs, photo_masks, tm_folder):
    transf_edge_coords_list = []
    transf_mask_coords_list = []
    
    for photo_nr, photo_mask in zip(photo_nrs, photo_masks):
        edge_coords, mask_coords = get_photo_coords(photo_mask)
        tm = ParallelSliceSimilarityTM(path = tm_folder / f"{scan_number}" / f"registration_params_{scan_number}_{photo_nr}.txt")
        #tm = ParallelSliceSimilarityTM(path = 
        #    get_results_folder() / "2023-04-07_register_kanzi_dataset_MSE_overlap_1" / "registration_parameters" / f"registration_params_{scan_number}.txt")
        
        transf_edge_coords_list.append(tm.transform([photo_nr], [edge_coords])[0])
        transf_mask_coords_list.append(tm.transform([photo_nr], [mask_coords])[0])
        
    return transf_edge_coords_list, transf_mask_coords_list

def any_in_hull(p, hull):
    """
    From: https://stackoverflow.com/questions/16750618/whats-an-efficient-way-to-find-if-a-point-lies-in-the-convex-hull-of-a-point-cl
    Test if points in `p` are in `hull`

    `p` should be a `NxK` coordinates of `N` points in `K` dimensions
    `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the 
    coordinates of `M` points in `K`dimensions for which Delaunay triangulation
    will be computed
    """
    if not isinstance(hull,Delaunay):
        hull = Delaunay(hull)

    return np.any(hull.find_simplex(p)>=0)

if __name__ == "__main__":
    scans_folder = get_data_folder()
    photo_metadata_path = scans_folder / "photo_metadata.csv"
    bbox_data_path = scans_folder / "slice_photos" / "photo_bboxes.csv"
    # output folders
    experiment_folder = ceu.make_new_experiment_folder(get_results_folder())
    tm_folder = get_results_folder() / "2023-04-10_register_kanzi_dataset_separate_full" / "registration_parameters"


    bbox_data = BboxDataParser(bbox_data_path, full_size=(1942, 2590), edge_clamp_range=50)
    metadata_dict = parse_metadata(photo_metadata_path)
    
    with open(experiment_folder / "intersection_results.csv", "w") as out_file:
        out_file.write("scan_nr,intersected,intersection_nrs\n")
        for scan_number in tqdm(range(9, 121)):
            scan_name = str(scan_number)
            ct_mask_path = scans_folder / "masks2" / scan_name
            photo_path = scans_folder / "slice_photos_crop" / scan_name
            photo_mask_path = scans_folder / "slice_photo_masks_crop_all" / scan_name
            
            metadata = metadata_dict[scan_number]
            
            if not metadata.included:
                continue

            photo_nrs, photo_masks, _ = load_photo_data(photo_path, photo_mask_path, scan_name, metadata.excluded_slices)
            
            transf_edge_coords_list, transf_mask_coords_list = get_photo_coords_lists(scan_number, photo_nrs, photo_masks, tm_folder)
            
            intersected = False
            intersection_nrs = []
            for i, photo_nr in enumerate(photo_nrs):
                if i > 1:
                    hull_points = np.concatenate(transf_edge_coords_list[:i])
                    if any_in_hull(transf_mask_coords_list[i], hull_points):
                        intersected = True
                        intersection_nrs.append(photo_nr)
                        continue
                if i < len(photo_nrs)-3:
                    hull_points = np.concatenate(transf_edge_coords_list[i+1:])
                    if any_in_hull(transf_mask_coords_list[i], hull_points):
                        intersected = True
                        intersection_nrs.append(photo_nr)
            out_str = f"{scan_number},{int(intersected)},{' '.join([str(x) for x in intersection_nrs])}"
            print(out_str)
            out_file.write(out_str + "\n")
