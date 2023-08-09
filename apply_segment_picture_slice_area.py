import torch
import numpy as np
from pathlib import Path
from folder_locations import get_data_folder
from segment_picture_slice_area_MSD import LightningMSDSegmentation
from tqdm import tqdm
import ct_experiment_utils as ceu
from matplotlib import pyplot as plt
import skimage.io
import re
import random
import scipy
import tifffile

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
        
def apply_and_save_raw(photos_path, masks_out_path, indices, model):
    for apple, picture in tqdm(indices):
        print((apple, picture))
        photo_path = photos_path / f"{apple}" / f"Kanzi{apple}_slice_{picture}.png"
        mask_path = masks_out_path / f"{apple}" / f"Kanzi{apple}_slice_{picture}.tif"
        
        photo = np.moveaxis(skimage.io.imread(photo_path), 2, 0).astype(np.float32)/255
        inphoto = photo - np.array((0.65979585, 0.51474652, 0.34878069), dtype=np.float32)[:, None, None]
        inphoto /= np.array((0.32747177, 0.28363168, 0.17863797), dtype=np.float32)[:, None, None]
        inphoto = torch.from_numpy(inphoto).cuda()
        
        mask_out = model(inphoto[None, ...]).detach().cpu()[0, 0, :, :].numpy()   
             
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(str(mask_path), mask_out)

if __name__ == '__main__':
    experiment_folder = Path(__file__).resolve().parents[1]
    scans_folder = get_data_folder()
    photos_path = scans_folder / "slice_photos_crop"
    masks_path = scans_folder / "slice_photo_masks_crop"
    masks_out_path = experiment_folder / "slice_photo_masks_crop_nn_output"
    
    checkpoint = next((experiment_folder / "checkpoints" / "validation_loss").glob("best_validation_loss_epoch=*.ckpt"))
    
    model = LightningMSDSegmentation.load_from_checkpoint(str(checkpoint))
    model.freeze()
    model.cuda()
    
    excluded_indices = read_photo_metadata(scans_folder / "photo_metadata.csv")
    
    picture_indices = []

    for i in range(9, 121):
        files = list((photos_path / f"{i}").glob("Kanzi*_slice_*.png"))
        for file in files:
            slice_number = int(re.findall('\d+', file.name)[-1])
            index = (i, slice_number)
            if not index in excluded_indices:
                picture_indices.append(index)

    #print(f"Segmenting {len(picture_indices)} pictures")
    apply_and_save_raw(photos_path, masks_out_path, picture_indices, model)
