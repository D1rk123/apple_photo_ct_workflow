import re
import numpy as np
import torch

def angles_xyz_to_rotation_matrix(angles_xyz):
    dev = angles_xyz.device
    zero = torch.tensor(0, dtype=torch.float32, device=dev)
    one = torch.tensor(1, dtype=torch.float32, device=dev)
    
    rot_matrix_x = torch.stack([
        torch.stack([one, zero, zero]),
        torch.stack([zero, torch.cos(angles_xyz[0]), -torch.sin(angles_xyz[0])]),
        torch.stack([zero, torch.sin(angles_xyz[0]), torch.cos(angles_xyz[0])])
    ])
    rot_matrix_y = torch.stack([
        torch.stack([torch.cos(angles_xyz[1]), zero, torch.sin(angles_xyz[1])]),
        torch.stack([zero, one, zero]),
        torch.stack([-torch.sin(angles_xyz[1]), zero, torch.cos(angles_xyz[1])])
    ])
    rot_matrix_z = torch.stack([
        torch.stack([torch.cos(angles_xyz[2]), -torch.sin(angles_xyz[2]), zero]),
        torch.stack([torch.sin(angles_xyz[2]), torch.cos(angles_xyz[2]), zero]),
        torch.stack([zero, zero, one])
    ])
    return torch.mm(torch.mm(rot_matrix_z, rot_matrix_y), rot_matrix_x)


class ParallelSliceSimilarityTM():
    def __init__(self, slice_nrs=None, path=None):
        if slice_nrs is not None and path is None:
            self.init_zeros(slice_nrs)
        elif slice_nrs is None and path is not None:
            self.read(path)
        elif slice_nrs is None and path is None:
            raise Exception("Wrong initilization: Provide either slice_nrs or path")
        else:
            raise Exception("Wrong initilization: Don't provide both slice_nrs and path")

            
    def init_zeros(self, slice_nrs):
        self.slice_nrs = slice_nrs
        self.angles_xyz = torch.zeros((3, ), dtype=torch.float32)
        self.slice_thickness = torch.zeros((1, ), dtype=torch.float32)
        self.slice_offset = torch.zeros((1, ), dtype=torch.float32)
        self.scale = torch.ones((1, ), dtype=torch.float32)
        self.slice_xy_offsets = {}
        for slice_nr in slice_nrs:
            self.slice_xy_offsets[slice_nr] = torch.zeros((1, ), dtype=torch.float32)
        
    def write(self, path):
        with open(path, "w") as regparams_file:
            regparams_file.write(f"{self.angles_xyz[0]:.30f}"+"\n")
            regparams_file.write(f"{self.angles_xyz[1]:.30f}"+"\n")
            regparams_file.write(f"{self.angles_xyz[2]:.30f}"+"\n")
            regparams_file.write(f"{self.slice_thickness:.30f}"+"\n")
            regparams_file.write(f"{self.slice_offset:.30f}"+"\n")
            regparams_file.write(f"{self.scale:.30f}"+"\n")
            for slice_nr in self.slice_nrs:
                offset = self.slice_xy_offsets[slice_nr]
                regparams_file.write(f"{slice_nr}:{offset[0]:.30f},{offset[1]:.30f}\n")
                
    def read(self, path):
        with open(path, "r") as regparams_file:
            lines = regparams_file.readlines()

        self.angles_xyz = torch.tensor([float(lines[0]), float(lines[1]), float(lines[2])], dtype=torch.float32)
        self.slice_thickness = torch.tensor(float(lines[3]), dtype=torch.float32)
        self.slice_offset = torch.tensor(float(lines[4]), dtype=torch.float32)
        self.scale = torch.tensor(float(lines[5]), dtype=torch.float32)
        
        self.slice_nrs = []
        self.slice_xy_offsets = {}
        for line in lines[6:]:
            line_parts = re.split("[:,]", line)
            slice_nr = int(line_parts[0])
            self.slice_nrs.append(slice_nr)
            self.slice_xy_offsets[slice_nr] = torch.tensor([float(line_parts[1]), float(line_parts[2])], dtype=torch.float32)
        
            
    def transform(self, slice_nrs, slice_coords_list):
        transformed_coords_list = []
        rotation_matrix = angles_xyz_to_rotation_matrix(self.angles_xyz)
        
        for slice_nr, slice_coords in zip(slice_nrs, slice_coords_list):
            z_coord = -slice_nr*self.slice_thickness + self.slice_offset

            out_coords = slice_coords * self.scale
            out_coords[:, 0:2] += self.slice_xy_offsets[slice_nr]
            out_coords[:, 2] += z_coord
            transformed_coords_list.append(torch.mm(out_coords, rotation_matrix))
        
        return transformed_coords_list
        
    def require_grad_(self):
        self.angles_xyz.requires_grad_()
        if len(self.slice_nrs) > 1:
            self.slice_thickness.requires_grad_() 
        self.slice_offset.requires_grad_()
        self.scale.requires_grad_()
        for i in self.slice_xy_offsets:
            self.slice_xy_offsets[i].requires_grad_()
            
    def detach_(self):
        self.angles_xyz.detach_()
        self.slice_thickness.detach_() 
        self.slice_offset.detach_()
        self.scale.detach_()
        for i in self.slice_xy_offsets:
            self.slice_xy_offsets[i].detach_()
            
    def to_(self, device):
        self.angles_xyz = self.angles_xyz.to(device)
        self.slice_thickness = self.slice_thickness.to(device)
        self.slice_offset = self.slice_offset.to(device)
        self.scale = self.scale.to(device)
        for i in self.slice_xy_offsets:
            self.slice_xy_offsets[i] = self.slice_xy_offsets[i].to(device)
            
    def take_subset(self, slice_nrs_subset):
        subset_tm = ParallelSliceSimilarityTM(slice_nrs_subset)
        subset_tm.angles_xyz = torch.clone(self.angles_xyz)
        subset_tm.slice_thickness = torch.clone(self.slice_thickness)
        subset_tm.slice_offset = torch.clone(self.slice_offset)
        subset_tm.scale = torch.clone(self.scale)
        subset_tm.slice_xy_offsets = {}
        for slice_nr in slice_nrs_subset:
            subset_tm.slice_xy_offsets[slice_nr] = torch.clone(self.slice_xy_offsets[slice_nr])
        return subset_tm
        
    def get_differentiable_params(self):
        if len(self.slice_nrs) == 1:
            params = [self.angles_xyz, self.slice_offset, self.scale]
        else:
            params = [self.angles_xyz, self.slice_thickness, self.slice_offset, self.scale]
        params += self.slice_xy_offsets.values()
        return params
        
    def print_with_grad(self):
        print(f"angles_xyz = {self.angles_xyz.detach().cpu().numpy()},   self.angles_xyz.grad = {self.angles_xyz.grad.detach().cpu().numpy()}")
        if len(self.slice_nrs) > 1:
            print(f"slice_thickness = {self.slice_thickness.detach().cpu().numpy()},   slice_thickness.grad = {self.slice_thickness.grad.detach().cpu().numpy()}")
        print(f"slice_offset = {self.slice_offset.detach().cpu().numpy()},   slice_offset.grad = {self.slice_offset.grad.detach().cpu().numpy()}")
        print(f"scale = {self.scale.detach().cpu().numpy()},   scale.grad = {self.scale.grad.detach().cpu().numpy()}")
        for slice_nr in self.slice_nrs:
            print((f"slice_xy_offsets[{slice_nr}] = {self.slice_xy_offsets[slice_nr].detach().cpu().numpy()},"
            f"   slice_xy_offsets[{slice_nr}].grad = {self.slice_xy_offsets[slice_nr].grad.detach().cpu().numpy()}"))
