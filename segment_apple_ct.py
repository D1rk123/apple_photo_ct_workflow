from pathlib import Path
import numpy as np
from skimage.filters import threshold_otsu
from skimage.measure import label
from skimage.segmentation import flood
from skimage.morphology import binary_dilation, binary_closing, ball

import ct_experiment_utils as ceu
from folder_locations import get_results_folder, get_data_folder

def segment_apple(image, threshold):
    #use otsu thresholding to segment the apple
    mask = image > threshold_otsu(image)
    
    # closing by 7 pixels
    mask = binary_closing(mask, ball(7))
    
    # find the largest connected component
    # https://stackoverflow.com/questions/47540926/get-the-largest-connected-component-of-segmentation-image
    labels = label(mask)
    assert( labels.max() != 0 ) # assume at least 1 CC
    mask = (labels == np.argmax(np.bincount(labels.flat)[1:])+1)
    
    # fill all holes within the connected component
    mask = (flood(mask, (0,0,0)) == False)
    
    return mask


if __name__ == '__main__':
    base_path = get_data_folder()
    recon_path = base_path / "fdk_bh_corrected_recons"
    out_path = base_path / "masks2"
    experiment_folder = ceu.make_new_experiment_folder(get_results_folder())
    
    thresholds = []
    for i in range(1, 121):
        image = ceu.load_stack(recon_path / f"{i}", prefix="")
        thresholds.append(threshold_otsu(image))
        print(thresholds[-1])
    print(f"min={np.min(thresholds)}, max={np.max(thresholds)}, mean={np.mean(thresholds)}")
    threshold = np.mean(thresholds)
    with open(experiment_folder / "threshold.txt", "w") as file:
        file.write(f"{threshold}")

    for i in range(1, 121):
        #print(recon_path / f"{i}")
        image = ceu.load_stack(recon_path / f"{i}", prefix="")
        mask = segment_apple(image, threshold)
        ceu.save_stack(out_path / f"{i}", mask)
