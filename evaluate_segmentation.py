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
from matplotlib import pyplot as plt

import ct_experiment_utils as ceu
from folder_locations import get_data_folder, get_results_folder
from combine_nn_outputs import postprocess_mask

def score_nn_segmentation(nn_seg, gt_seg):
    TP = np.sum(np.logical_and(nn_seg, gt_seg))
    FP = np.sum(np.logical_and(nn_seg, np.logical_not(gt_seg)))
    FN = np.sum(np.logical_and(np.logical_not(nn_seg), gt_seg))
    TN = np.sum(np.logical_and(np.logical_not(nn_seg), np.logical_not(gt_seg)))
    
    accuracy = (TP + TN) / (TP + TN + FN + FP)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2*TP / (2*TP + FP + FN)
    
    return accuracy, precision, recall, f1
    

if __name__ == '__main__':
    scans_folder = get_data_folder()
    test_masks_folder = scans_folder / "slice_photo_test_masks_crop"
    save_folder = get_results_folder()
    experiment_folder = ceu.make_new_experiment_folder(save_folder, name="evaluate_segmentation")

    nn_output_folders = [
        save_folder / "split0_2023-01-23_picture_slice_segmentation_1" / "slice_photo_masks_crop_nn_output",
        save_folder / "split1_2023-01-23_picture_slice_segmentation_1" / "slice_photo_masks_crop_nn_output",
        save_folder / "split2_2023-01-23_picture_slice_segmentation_1" / "slice_photo_masks_crop_nn_output",
        save_folder / "split3_2023-01-23_picture_slice_segmentation_1" / "slice_photo_masks_crop_nn_output",
        save_folder / "split4_2023-01-30_picture_slice_segmentation_1" / "slice_photo_masks_crop_nn_output",
        ]
        
    test_mask_indices = [(16, 1), (23, 3), (36, 3), (41, 4), (52, 6), (53, 6), (55, 7), (59, 8), (61, 9), (72, 10), (75, 16), (86, 1), (89, 13), (93, 14), (104, 2), (109, 2), (111, 5), (112, 5), (118, 7), (120, 9)]
    
    results_all = [[] for _ in range(len(nn_output_folders)+1)]
    
    for scan_nr, photo_nr in test_mask_indices:
        gt_mask_path = test_masks_folder / f"{scan_nr}" / f"Kanzi{scan_nr}_slice_{photo_nr}.png"
        gt_seg = skimage.io.imread(gt_mask_path)[:, :, 0] < 128
        
        gt_seg_edge = gt_seg ^ skimage.morphology.binary_erosion(gt_seg)
        gt_dm = scipy.ndimage.distance_transform_edt(np.logical_not(gt_seg_edge))
        
        nn_outputs = []
        for folder in nn_output_folders:
            output_path = folder / f"{scan_nr}" / f"Kanzi{scan_nr}_slice_{photo_nr}.tif"
            nn_outputs.append(tifffile.imread(output_path))
        nn_outputs.append(np.mean(np.stack(nn_outputs, axis=0), axis=0))
        
        for nn_output, results in zip(nn_outputs, results_all):
            nn_seg = postprocess_mask(nn_output, photo_nr>3)
            accuracy, precision, recall, f1 = score_nn_segmentation(gt_seg, nn_seg)

            nn_edge_distances = gt_dm[nn_seg ^ skimage.morphology.binary_erosion(nn_seg)]
            mean_edge_dist = np.mean(nn_edge_distances)
            mean_edge_dist_mm = mean_edge_dist * ((129.301748*0.3637134)/1000) # rough conversion from voxels to mm based on voxel size and registered scale parameter
            results.append([scan_nr,photo_nr,accuracy,precision,recall,f1,mean_edge_dist,mean_edge_dist_mm])
    
    csv_header = "scan_nr,photo_nr,accuracy,precision,recall,f1,mean_edge_dist,mean_edge_dist_mm"
    names = ["split0", "split1", "split2", "split3", "split4", "averaged"]        
    for results, name in zip(results_all, names):
        with open(experiment_folder / (name+".csv"), "w") as file:
            file.write(csv_header + "\n")
            for result in results:
                file.write(",".join([str(r) for r in result]) + "\n")

    means = []
    stddevs = []
    with open(experiment_folder / "latex_table.txt", "w") as file:
        for results, name in zip(results_all, names):
            file.write(f"{name}")
            for i in [2, 3, 4, 7]:
                scores = np.array([r[i] for r in results])
                file.write(f" & ${np.mean(scores):.4f} (\pm {np.std(scores):.4f})$")
            file.write(" \\\\ \n")
            
    
    
        
