import torch
import numpy as np
import scipy
import skimage

from image_coordinate_utils import volume_lookup

class EdgeL1DistanceCost():
    def _make_distance_map(self, volume_mask):
        np_volume_mask = volume_mask.numpy()
        edge = volume_mask ^ skimage.morphology.binary_erosion(volume_mask)
        distance_volume = scipy.ndimage.distance_transform_edt(np.logical_not(edge)).astype(np.float32)
        return torch.from_numpy(distance_volume)

    def __init__(self, slice_nrs, slice_edge_coords_list, volume_mask):
        self.slice_nrs = slice_nrs
        self.slice_edge_coords_list = slice_edge_coords_list
        self.distance_volume = self._make_distance_map(volume_mask)
        
    def evaluate(self, tm):
        total_error = torch.tensor(0)
        transformed_coords_list = tm.transform(self.slice_nrs, self.slice_edge_coords_list)
        num_coords = sum([tc.size(dim=0) for tc in transformed_coords_list])
        
        for transformed_coords in transformed_coords_list:
            dm_lookup = volume_lookup(self.distance_volume, transformed_coords).to(dtype=torch.float64)
            slice_error = torch.sum(dm_lookup)
            slice_error = slice_error / num_coords
            total_error = total_error + slice_error
            
        return total_error
        
    def to_(self, device):
        for i in range(len(self.slice_edge_coords_list)):
            self.slice_edge_coords_list[i] = self.slice_edge_coords_list[i].to(device)
        self.distance_volume = self.distance_volume.to(device)
        
class OverlapMseCost():
    def __init__(self, slice_nrs, slice_coords_list, slice_mask_values_list, volume_mask):
        self.slice_nrs = slice_nrs
        self.slice_coords_list = slice_coords_list
        self.slice_mask_values_list = slice_mask_values_list
        self.volume_mask = volume_mask.to(dtype=torch.float32)
        
    def evaluate(self, tm):
        total_error = torch.tensor(0)
        transformed_coords_list = tm.transform(self.slice_nrs, self.slice_coords_list)
        num_coords = sum([tc.size(dim=0) for tc in transformed_coords_list])
        
        for transformed_coords, slice_mask_values in zip(transformed_coords_list, self.slice_mask_values_list):
            mask_lookup = volume_lookup(self.volume_mask, transformed_coords)
            slice_error = torch.sum((mask_lookup - slice_mask_values.to(torch.float32))**2)
            slice_error = slice_error / num_coords
            total_error = total_error + slice_error
            
        return total_error
        
    def to_(self, device):
        for i in range(len(self.slice_coords_list)):
            self.slice_coords_list[i] = self.slice_coords_list[i].to(device)
            self.slice_mask_values_list[i] = self.slice_mask_values_list[i].to(device)
        self.volume_mask = self.volume_mask.to(device)
        
